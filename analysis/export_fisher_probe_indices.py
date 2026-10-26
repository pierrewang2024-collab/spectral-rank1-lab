# ============================================================================
# 文件名：analysis/export_fisher_probe_indices.py
# 职责：从 WebNLG 训练样本窗口估计 Empirical Fisher（|FFT2(∇_W L)|² 的 batch 均值），
#       导出 FourierFT 坐标语义的频点字典 Dict[full_layer_name, LongTensor[2,K]]。
# 协议：GPT-2-Medium / WebNLG / FP32 / eval / batch=8 / 96 target layers。
# 重要：默认不排除 DC，与参考选点实现的语义保持一致。
# 输出：<output_dir>/fisher_indices.pt + fisher_probe_metadata.json（可审计样本 ID 等）
#
# 教学版适配（2026-10-26）：
#   * 依赖改为本目录同侪导入（freq_fisher_heatmap / data_loaders / utils_device）；
#   * 默认模型从服务器本地路径改为 HF hub id "gpt2-medium"（首次运行自动下载 ~1.4GB）；
#   * 移除服务器离线缓存环境变量（发布环境需联网）。
# 逻辑与原版完全一致，未做任何算法性改动。
# ============================================================================
"""导出独立 Fisher probe window 的 FourierFT indices（发布版）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from data_loaders import load_task  # noqa: E402
from freq_fisher_heatmap import compute_freq_fisher  # noqa: E402
from utils_device import describe, get_device  # noqa: E402

DEFAULT_MODEL = "gpt2-medium"
TARGET_MODULES = ["c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]


def _matched_target_layers(model):
    """与参考实现完全相同的精确层匹配。"""
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:
        Conv1D = ()

    matched, seen = [], set()
    for name, module in model.named_modules():
        if name in seen or not any(name.endswith(t) for t in TARGET_MODULES):
            continue
        if not isinstance(module, (torch.nn.Linear, Conv1D)):
            continue
        seen.add(name)
        matched.append((name, module, isinstance(module, Conv1D)))
    if not matched:
        raise RuntimeError("未匹配到任何目标层")
    return matched


def _select_topk(heatmap: torch.Tensor, k: int, exclude_dc: bool) -> torch.Tensor:
    """返回 FourierFT 坐标语义的 [2,K] LongTensor。"""
    flat = heatmap.flatten().clone()
    if exclude_dc:
        flat[0] = -float("inf")
    if k > flat.numel() - int(exclude_dc):
        raise ValueError(f"k={k} 超过该层可选频点数 {flat.numel() - int(exclude_dc)}")
    top = torch.topk(flat, k=k, largest=True).indices
    width = heatmap.shape[1]
    return torch.stack([(top // width).long(), (top % width).long()], dim=0).cpu()


def _choose_sample_ids(n_train: int, n_samples: int, mode: str, seed: int,
                       start_batch: int, batch_size: int) -> list[int]:
    if n_samples > n_train:
        raise ValueError(f"请求 {n_samples} 条样本，但 train 只有 {n_train} 条")
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(n_train, generator=g)[:n_samples].tolist()
    start = start_batch * batch_size
    stop = start + n_samples
    if stop > n_train:
        raise ValueError(f"offset 窗口 [{start}:{stop}) 超出训练集长度 {n_train}")
    return list(range(start, stop))


def main() -> None:
    p = argparse.ArgumentParser(description="导出独立 Fisher probe window 的频点")
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--n_frequency", type=int, required=True)
    p.add_argument("--probe_mode", choices=("random", "offset"), default="random")
    p.add_argument("--probe_seed", type=int, default=20260825,
                   help="仅 random 模式使用；独立于训练 seed 与频点随机 seed")
    p.add_argument("--start_batch", type=int, default=8,
                   help="仅 offset 模式使用；默认 [8:16)，与当前 [0:8) 协议窗口不重叠")
    p.add_argument("--probe_batch_size", type=int, default=8)
    p.add_argument("--fisher_n_batches", type=int, default=8)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--exclude_dc", action="store_true",
                   help="仅在将来主训练也启用排 DC 时使用；对照协议下请勿传此项")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    if args.probe_batch_size != 8 or args.fisher_n_batches != 8:
        print("[WARN] 非主协议 probe 设置；稳健性对照应固定 batch_size=8、n_batches=8")
    n_samples = args.probe_batch_size * args.fisher_n_batches
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from torch.utils.data import DataLoader, Subset
    from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

    device = get_device()
    print(f"[device] {describe()}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    train_ds, *_ = load_task(
        {"kind": "nlg", "dataset": "web_nlg", "max_length": args.max_length}, tokenizer,
    )
    sample_ids = _choose_sample_ids(
        len(train_ds), n_samples, args.probe_mode, args.probe_seed,
        args.start_batch, args.probe_batch_size,
    )
    if len(set(sample_ids)) != n_samples:
        raise RuntimeError("probe sample IDs 出现重复")
    protocol_ids = set(range(n_samples))
    overlap = len(protocol_ids.intersection(sample_ids))
    subset = Subset(train_ds, sample_ids)
    loader = DataLoader(subset, batch_size=args.probe_batch_size, shuffle=False,
                        collate_fn=default_data_collator)
    print(f"[probe] mode={args.probe_mode} n_samples={n_samples} "
          f"seed={args.probe_seed if args.probe_mode == 'random' else 'N/A'} "
          f"overlap_with_default_head64={overlap}")

    matched = _matched_target_layers(model)
    layer_names = [name for name, _, _ in matched]
    conv1d_flags = {name: is_conv1d for name, _, is_conv1d in matched}
    print(f"[layers] matched={len(layer_names)}")

    heatmaps = compute_freq_fisher(
        model, loader, layer_names, n_batches=args.fisher_n_batches,
        device=device, dtype=torch.float32, verbose=True,
    )
    indices = {}
    for name in layer_names:
        heat = heatmaps[name]
        # 注意：Conv1D 权重需从 [in,out] 转为 [out,in] 再选点。
        if conv1d_flags[name]:
            heat = heat.T.contiguous()
        indices[name] = _select_topk(heat, args.n_frequency, args.exclude_dc)

    out_pt = out_dir / "fisher_indices.pt"
    torch.save(indices, out_pt)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "independent Fisher probe-window robustness full-training control",
        "model": args.model_name,
        "dataset": "web_nlg/release_v3.0_en train",
        "train_size": len(train_ds),
        "probe_mode": args.probe_mode,
        "probe_seed": args.probe_seed if args.probe_mode == "random" else None,
        "start_batch": args.start_batch if args.probe_mode == "offset" else None,
        "probe_batch_size": args.probe_batch_size,
        "fisher_n_batches": args.fisher_n_batches,
        "n_samples": n_samples,
        "sample_ids": sample_ids,
        "overlap_with_default_head64": overlap,
        "shuffle": False,
        "model_dtype": "fp32",
        "model_mode": "eval",
        "target_modules": TARGET_MODULES,
        "n_layers": len(indices),
        "n_frequency_per_layer": args.n_frequency,
        "total_trainable_spectral_params": len(indices) * args.n_frequency,
        "exclude_dc": args.exclude_dc,
        "selection_semantics": "strict top-k empirical Fisher; Conv1D heatmaps transposed to [out,in]",
        "indices_file": str(out_pt.resolve()),
        "wall_time_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "fisher_probe_metadata.json"
    out_json.write_text(json.dumps(metadata, indent=2))
    print(f"[done] indices={out_pt} layers={len(indices)}")
    print(f"[done] metadata={out_json} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
