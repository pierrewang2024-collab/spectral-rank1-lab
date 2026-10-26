#!/usr/bin/env python3
"""
谱分析脚本（v3-2026-10-01 移植，教学版）：Fisher(梯度选点) vs Vanilla(随机选点)
训后 ΔW 的谱结构。相对研究原型的唯一改动：
  * 根目录 ROOT 支持 --root 参数覆盖（默认 = 本脚本所在目录的上一级，即仓库根），
    便于把 checkpoint 放在任意位置后运行。

原版 v3 要点（保持不变）：
  * 按真实层尺寸 (out_features, in_features) 重建 ΔW = Re(ifft2(dense_spectrum))·scaling，
    与训练时的 ΔW 重建路径一致（v2 曾用 indices.max()+1
    推断 FFT 网格，GPT-2 Fisher mlp.c_proj 选点全在谱第 0 列时会错推断成 1024×1——
    这是已知陷阱，v3 已修）。scaling 为常数，不影响有效秩与归一化熵。
  * 精确奇异值：Gram 矩阵特征值求完整奇异值谱（float64）。
  * 选点支撑结构：distinct rows R / distinct cols Q / 结构秩上界 2·min(R,Q) /
    共轭成对比例（实系数参数化下 q 与 -q 对应同一余弦原子）。
  * GPT-2 全部 24 层；Llama 每 4 层取 1 层。无 Llama checkpoint 时自动跳过（MISS）。
  * 不依赖 torch：zip+pickle 直接读 torch checkpoint（只需 numpy）。

运行（任意目录均可，只需 numpy）：
  python3 analysis/spectral_energy_analysis.py --smoke   # 自检：只算第 0 层
  python3 analysis/spectral_energy_analysis.py            # 全量（GPT-2 部分）
  python3 analysis/spectral_energy_analysis.py --root /path/containing/results
输出：同目录 spectral_energy.json（--smoke 时为 spectral_energy_smoke.json）
checkpoint 放置：在 ROOT 下建 results/ 目录，目录名需匹配 RUNS 的 glob 模式，
例如 results/FISHER96K_B64_myrun__evalfix_main_fish96k_s42/final_adapter.pt。
"""
import collections
import glob
import json
import os
import pickle
import re
import sys
import time
import zipfile

import numpy as np

SMOKE = "--smoke" in sys.argv


def _arg_value(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            return sys.argv[i + 1]
    return default


ROOT = os.path.abspath(_arg_value(
    "--root", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

# 真实层尺寸 (out_features, in_features)，与 FourierFT dense_spectrum 的行列一致。
# GPT-2 Medium: n_embd=1024, n_inner=4096（Conv1D 权重存为 (in, out)，层内已换算为 out×in）。
# Llama-3-8B: hidden=4096, intermediate=14336, num_key_value_heads=8, head_dim=128。
SHAPES = {
    "gpt2": {"attn.c_attn": (3072, 1024), "attn.c_proj": (1024, 1024),
             "mlp.c_fc": (4096, 1024), "mlp.c_proj": (1024, 4096)},
    "llama": {"q_proj": (4096, 4096), "k_proj": (1024, 4096), "v_proj": (1024, 4096),
              "o_proj": (4096, 4096), "gate_proj": (14336, 4096), "up_proj": (14336, 4096),
              "down_proj": (4096, 14336)},
}
LAYERS = {"gpt2": set(range(24)), "llama": set(range(0, 32, 4))}
if SMOKE:
    LAYERS = {"gpt2": {0}, "llama": {0}}

RUNS = {
    "gpt2": {
        "Fisher":  "results/FISHER96K_B64_*__evalfix_main_fish96k_s42/final_adapter.pt",
        "Vanilla": "results/VANILLA96K_B64_*__evalfix_main_van96k_s42/final_adapter.pt",
    },
    "llama": {
        "Fisher":  "results/LLAMA3_FISHER426K_*__llama3_lrgrid_0901_fisher_lr1e-2_s42/final_adapter.pt",
        "Vanilla": "results/LLAMA3_VANILLA426K_*__llama3_lrgrid_0901_vanilla_lr1e-2_s42/final_adapter.pt",
    },
}


# ---------------------------------------------------------------- checkpoint reader（无 torch）
_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
       "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
       "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
       "BFloat16Storage": "bf16"}


class _Storage:
    def __init__(self, name):
        self.name = name


def _rebuild(storage, offset, size, stride, *rest):
    if len(size) == 0:
        return storage[offset].copy()
    it = storage.itemsize
    return np.array(np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * it for s in stride)))


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix, self.cache = zf, prefix, {}

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a: data
        if module == "torch" and name.endswith("Storage"):
            return _Storage(name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch":
            return name
        return super().find_class(module, name)

    def persistent_load(self, pid):
        _, stype, key, _loc, _n = pid
        if key not in self.cache:
            raw = self.zf.read(f"{self.prefix}/data/{key}")
            dt = _DT[stype.name]
            self.cache[key] = ((np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
                               if dt == "bf16" else np.frombuffer(raw, dt))
        return self.cache[key]


def load_ckpt(path):
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
    with zf.open(pkl) as f:
        return _Unpickler(f, zf, pkl[: -len("/data.pkl")]).load()


# ---------------------------------------------------------------- metrics
def norm_entropy(x):
    p = np.abs(np.asarray(x)).astype(np.float64).ravel()  # 先取模再转实数，复数输入不丢虚部
    p = p / (p.sum() + 1e-300)
    p = p[p > 1e-10]
    return float(-np.sum(p * np.log(p)) / np.log(len(p))) if len(p) > 1 else 0.0


def singular_values(w):
    """完整奇异值谱：Gram 矩阵（较小一侧）特征值开方，float64。"""
    w = np.ascontiguousarray(w, dtype=np.float64)  # .real 视图非连续，会让 matmul 退出 BLAS 快路径
    g = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
    ev = np.linalg.eigvalsh(g)
    return np.sqrt(np.clip(ev[::-1], 0.0, None))


def eff_rank(s, rel_tol=1e-6):
    s = s[s > rel_tol * s[0]] if s[0] > 0 else s[:0]
    if len(s) == 0:
        return 0.0, 0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p)))), int(len(s))


def module_type(scale, mod):
    for key in SHAPES[scale]:
        if mod.endswith(key):
            return key
    raise KeyError(mod)


def layer_id(mod):
    m = re.search(r"\.(\d+)\.", mod)
    return int(m.group(1)) if m else -1


def analyze_fourierft(scale, path):
    raw = load_ckpt(path)
    st, idx_map = raw["adapter_state_dict"], raw["indices_by_module"]
    rows = []
    for k, sp in st.items():
        if "fourierft_spectrum" not in k:
            continue
        mod = k.split(".fourierft_spectrum")[0]
        if layer_id(mod) not in LAYERS[scale]:
            continue
        t0 = time.time()
        t = module_type(scale, mod)
        out_f, in_f = SHAPES[scale][t]
        ind = np.asarray(next(iter(idx_map[mod].values())), dtype=np.int64)
        assert ind[0].max() < out_f and ind[1].max() < in_f, f"indices 越界 {mod}"
        sp = np.asarray(sp, dtype=np.float64).ravel()
        dense = np.zeros((out_f, in_f), dtype=np.complex128)
        dense[ind[0], ind[1]] = sp
        dw = np.ascontiguousarray(np.fft.ifft2(dense).real)
        del dense
        s = singular_values(dw)
        er, nrank = eff_rank(s)
        uniq = set(zip(ind[0].tolist(), ind[1].tolist()))
        conj = sum(((-u) % out_f, (-v) % in_f) in uniq and ((-u) % out_f, (-v) % in_f) != (u, v)
                   for u, v in uniq)
        R, Q = len(set(ind[0].tolist())), len(set(ind[1].tolist()))
        rows.append({
            "module": mod, "type": t, "layer": layer_id(mod), "out": out_f, "in": in_f,
            "n_freq": int(sp.size), "distinct_rows": R, "distinct_cols": Q,
            "structural_rank_bound": int(min(2 * min(R, Q), min(out_f, in_f))),
            "conjugate_paired_frac": conj / max(len(uniq), 1),
            "effective_rank": er, "numerical_rank": nrank,
            "effective_rank_top256": eff_rank(s[:256])[0],
            "spec_entropy_2d": norm_entropy(np.fft.fft2(dw)),
            "dw_fro_unscaled": float(np.linalg.norm(dw)),
        })
        print(f"  [{scale}] {t:12s} L{layer_id(mod):02d} eff_rank={er:8.2f} num_rank={nrank:5d} "
              f"R={R:5d} Q={Q:5d} conj={rows[-1]['conjugate_paired_frac']:.3f} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return rows


AGG_FIELDS = ("effective_rank", "numerical_rank", "effective_rank_top256", "spec_entropy_2d",
              "distinct_rows", "distinct_cols", "structural_rank_bound", "conjugate_paired_frac",
              "dw_fro_unscaled")

out = {"_meta": {"version": "v3-2026-10-01 (release-port)", "grid": "true layer shape (out_features, in_features)",
                 "singular_values": "exact, full spectrum (float64 Gram eigenvalues)",
                 "layers": {k: sorted(v) for k, v in LAYERS.items()}, "seed": 42,
                 "root": ROOT}}
for scale, methods in RUNS.items():
    out[scale] = {}
    for name, pat in methods.items():
        hits = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not hits:
            print(f"[{scale}] {name}: MISS ({pat})")
            continue
        print(f"[{scale}] {name}: {hits[-1]}", flush=True)
        rows = analyze_fourierft(scale, hits[-1])
        by_type = collections.OrderedDict()
        for r in rows:
            by_type.setdefault(r["type"], []).append(r)
        agg = {t: {"n_layers": len(rs), "n_freq": rs[0]["n_freq"], "out": rs[0]["out"], "in": rs[0]["in"],
                   **{f: float(np.mean([r[f] for r in rs])) for f in AGG_FIELDS}}
               for t, rs in by_type.items()}
        out[scale][name] = {"by_layer_type": agg, "n_modules": len(rows), "rows": rows}

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "spectral_energy_smoke.json" if SMOKE else "spectral_energy.json")
json.dump(out, open(dst, "w"), indent=1)
print("\nsaved:", dst)
print("DONE")
for scale in ("gpt2", "llama"):
    for name in ("Fisher", "Vanilla"):
        for t, a in out[scale].get(name, {}).get("by_layer_type", {}).items():
            print(f"{scale:6s} {name:8s} {t:12s} effR={a['effective_rank']:8.2f} numR={a['numerical_rank']:7.1f} "
                  f"bound={a['structural_rank_bound']:7.1f} R={a['distinct_rows']:7.1f} Q={a['distinct_cols']:7.1f} "
                  f"conj={a['conjugate_paired_frac']:.3f} ent2d={a['spec_entropy_2d']:.3f}")
