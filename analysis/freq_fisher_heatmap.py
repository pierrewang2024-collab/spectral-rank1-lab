# ============================================================================
# 文件名：analysis/freq_fisher_heatmap.py
# 职责：频域 Fisher 信息（heatmap）计算——频谱先验（zero-shot probe）核心。
#       对每层目标权重 W，给定下游小批量数据：
#         (1) 反传得 ∇_W L；
#         (2) 对 ∇_W L 做 2D-DFT；
#         (3) 取 |·|² 在 n_batches 上平均，得到该层每个频点的 Empirical Fisher。
#       该 heatmap 充当元启发式（GA-SA / PSO / CS）搜索初始化的先验。
# 输入：model（任意 nn.Module）、dataloader（输出 batch dict 或 (input_ids, labels)）、
#       target_layers（要分析的 nn.Linear 层名列表）、n_batches。
# 输出：dict[layer_name -> torch.Tensor of shape (out, in)]，存放 |F(∇W)|² 的均值。
# 上游依赖：torch；transformers / peft 仅在调用方使用
# 下游消费方：教学版仅保留探针本体（原研究仓库中还被搜索模块复用，与本课程无关）
# 版本：teaching-1.0（自研究版移植，算法逻辑未改动）
# 最后更新日期：2026-06-05
# 改编：课程组（源自内部研究代码）
#
# 复杂度备忘：
#   - GPT-2-Medium（d=1024）：单 batch FFT2 复杂度 O(d² log d) ≈ 1.0e7，n_batches=8
#     总耗时 < 60 s（CPU）/ < 5 s（GPU）。整模型 24 层 × 1 c_attn = 24 次 FFT2，可接受。
#   - LLaMA-3-8B（d=4096）：单 FFT2 ≈ 4096²·log(4096²) ≈ 4.0e8，
#     32 层 × 4 个 q/k/v/o = 128 次。建议使用 sparse_freq_fisher_chunked() 走分块 + fp16
#     路径，把每层切成 4×4 的 1024×1024 子块独立 FFT2 并直接累加幅度（牺牲少量谱泄漏换内存）。
# ============================================================================
"""
零次频域 Fisher 先验（教学版）。

理论锚点（可由 Fisher 信息在 ΔW→0 处的定义展开推出）：
    在 ΔW 取 0 附近，Fourier 参数 c_p 处的 Fisher 信息为
        I(c_p) = E[ |F(∇_W L)(u_p, v_p)| ² ]  + O(α)
    其中 α 是 FourierFT 的 scaling。本模块实现的就是该 Empirical Fisher 的样本估计。
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

# HuggingFace GPT-2 使用 transformers.pytorch_utils.Conv1D 作为"伪 Linear"
# （weight shape = [in, out]，与 nn.Linear 转置）。对频域 Fisher 计算而言，
# 维度语义不影响 2D-FFT |·|² 的计算（调用方以 weight.shape 为准）。
try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
    _LINEAR_LIKE = (nn.Linear, _HFConv1D)
except Exception:
    _LINEAR_LIKE = (nn.Linear,)


# ----------------------------------------------------------------------------
# § 标准实现（适用于 GPT-2-Medium 量级）
# ----------------------------------------------------------------------------
def _resolve_module(model: nn.Module, dotted_name: str) -> nn.Module:
    cur = model
    for tok in dotted_name.split("."):
        cur = getattr(cur, tok)
    return cur


def _to_loss(model_output, batch) -> torch.Tensor:
    """通用 loss 提取：HuggingFace 的 ModelOutput.loss 或手算 cross-entropy。"""
    if hasattr(model_output, "loss") and model_output.loss is not None:
        return model_output.loss
    logits = getattr(model_output, "logits", model_output)
    # 用 duck-typing 兼容 dict / BatchEncoding (UserDict 子类)
    labels = batch.get("labels") if hasattr(batch, "get") else None
    if labels is None:
        raise ValueError("batch 中缺少 labels 且 output 没有 loss 字段")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.enable_grad()
def compute_freq_fisher(
    model: nn.Module,
    dataloader: Iterable,
    target_layers: List[str],
    n_batches: int = 8,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
    grad_chunk_size: Optional[int] = None,
    use_gradient_checkpointing: bool = False,
) -> Dict[str, torch.Tensor]:
    """计算每个目标层的频域 Empirical Fisher heatmap。

    参数
    ----
    model:          已加载预训练权重的 nn.Module（可冻结也可不冻结，本函数只读 grad）
    dataloader:     可迭代 batch；每个 batch 为 dict（含 'input_ids' 等）或可送入 model 的对象
    target_layers:  要分析的层名（点分路径），通常是 nn.Linear 的名字
    n_batches:      用于平均的 batch 数；默认 8
    device:         模型/输入所在 device；默认沿用 model 当前 device
    dtype:          中间 FFT 计算 dtype；默认 fp32（fp16 在大模型上请用 sparse_freq_fisher_chunked）
    verbose:        打印进度
    grad_chunk_size: 2026-09-01 新增（8B OOM 修复）。默认 None=旧行为（单次全参数
                    backward，为所有 requires_grad 参数分配梯度——8B FP32 下
                    权重32GB+全模型梯度32GB 必然 OOM）。设置为正整数 N 时启用
                    分组反传：把 target 层按顺序切成每组 N 个矩阵，逐组
                    [冻结全部→解冻本组→逐batch forward+backward→收集梯度]，
                    每次只为本组参数分配梯度。数学上与单次全参数 backward
                    严格等价（各参数的 ∂L/∂W 相互独立），代价是前向次数变为
                    组数×n_batches（组数=ceil(len(target)/N)）。要求 dataloader
                    为顺序（shuffle=False），否则各组看到的样本不同破坏等价性。
    use_gradient_checkpointing: 2026-09-01 新增（probe 显存优化）。默认 False。
                    设为 True 时在函数进入时临时调用
                    model.gradient_checkpointing_enable()，probe 结束（finally）
                    调用 model.gradient_checkpointing_disable() 恢复原状。
                    gradient checkpointing 只是反向时用重计算代替缓存激活，
                    不改变梯度的数值（数学等价），而 Fisher 只读取 .grad，
                    因此启用后 Fisher 热力图完全不变；换来的是激活显存大幅
                    下降，允许 probe 用更大 batch（显著降低 896 次前反传的次数
                    或每组总耗时）。仅在 model 支持
                    gradient_checkpointing_enable 时生效（HF 模型均支持）。

    返回
    ----
    Dict[layer_name -> Tensor of shape (out_features, in_features)]
        每层的 |F(∇W)|² 在 n_batches 上的均值。
    """
    if device is None:
        device = next(model.parameters()).device

    # 2026-09-01：probe 显存优化——临时启用 gradient checkpointing（重计算激活代替
    # 缓存，梯度数值不变），probe 结束恢复原状。对整个函数作用域生效（旧/新两条
    # 反传路径共用同一前向）。
    _gc_enabled = False
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        _gc_enabled = True
        if verbose:
            print("[freq_fisher] gradient checkpointing ENABLED for probe "
                  "(activation recompute; grads numerically unchanged)")

    target_modules: Dict[str, nn.Module] = {}
    for name in target_layers:
        m = _resolve_module(model, name)
        if not isinstance(m, _LINEAR_LIKE):
            raise TypeError(
                f"target_layers 元素必须是 nn.Linear 或 transformers Conv1D，"
                f"但 {name} 是 {type(m)}"
            )
        if not hasattr(m, "weight"):
            raise TypeError(f"{name} 没有 weight 属性")
        target_modules[name] = m

    # 累加器（CPU 存放，避免 GPU OOM）——以 weight.shape 为准，兼容 Conv1D ([in,out]) / Linear ([out,in])
    heatmaps: Dict[str, torch.Tensor] = {
        n: torch.zeros(*m.weight.shape, dtype=dtype)
        for n, m in target_modules.items()
    }

    # 让目标层 .weight 至少 requires_grad 以便拿到 .grad
    for m in target_modules.values():
        m.weight.requires_grad_(True)

    def _forward_one_batch(batch):
        """单 batch 前向，返回 (loss, batch_dev)。duck-typing 兼容 BatchEncoding。"""
        if hasattr(batch, "items") and hasattr(batch, "keys"):
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            output = model(**batch_dev)
            return _to_loss(output, batch_dev), batch_dev
        elif isinstance(batch, (tuple, list)):
            batch_dev = tuple(v.to(device) if torch.is_tensor(v) else v for v in batch)
            output = model(*batch_dev)
            return _to_loss(output, {}), batch_dev
        raise TypeError(f"未知 batch 类型: {type(batch)}")

    def _accumulate_grad_ffts(group_modules):
        """对当前已有 .grad 的层，FFT2(grad) 取 |·|² 累加进 heatmaps。
        昆仑 910B 的 aclnnAbs 不支持 complex64，同时 aclnnFft 不支持 bf16。
        Fisher 仅 one-shot 计算，整个子过程走 CPU + fp32 是安全且可接受的。"""
        for name, m in group_modules.items():
            g = m.weight.grad
            if g is None:
                if verbose:
                    print(f"[WARN] layer {name} 反传后 grad 为空，跳过")
                continue
            g_cpu = g.detach().to("cpu", dtype=torch.float32)
            G = torch.fft.fft2(g_cpu)
            heatmaps[name] += (G.abs() ** 2).to(dtype)

    counted = 0
    t0 = time.time()

    if grad_chunk_size is None:
        # ---------- 旧行为：单次全参数 backward（GPT-2 规模安全） ----------
        for batch_idx, batch in enumerate(dataloader):
            if counted >= n_batches:
                break
            loss, _ = _forward_one_batch(batch)
            model.zero_grad(set_to_none=True)
            loss.backward()
            _accumulate_grad_ffts(target_modules)
            counted += 1
            if verbose:
                print(f"  [batch {counted}/{n_batches}] loss={float(loss.item()):.4f}  "
                      f"elapsed={time.time()-t0:.1f}s")
    else:
        # ---------- 2026-09-01 分组反传（8B OOM 修复，数学等价） ----------
        # 动机：单次 loss.backward() 会为所有 requires_grad 参数分配梯度。8B FP32 下
        # 权重 32GB + 全模型梯度 32GB > 61GB 上限，与 batch 大小无关（batch 只决定
        # 激活）。分组后每次 backward 只为本组参数分配梯度（Llama-8B 每组 8 矩阵
        # ≈1.2GB），峰值显存从 ~62GB 降到 ~36GB。
        # 等价性：链式法则下 ∂L/∂W_i 之间相互独立，逐组 backward 与一次全参数
        # backward 得到的每参数梯度逐值相同；Fisher 的 batch 均值不变。
        # 前提：dataloader 必须顺序（shuffle=False），否则各组看到不同样本。
        if getattr(dataloader, "shuffle", False):
            raise ValueError(
                "[freq_fisher] grad_chunk_size 模式要求顺序 dataloader（shuffle=False），"
                "否则各组 backward 看到的样本不同，破坏与全参数 backward 的等价性。"
            )
        chunk = max(1, int(grad_chunk_size))
        names = list(target_modules.keys())
        groups = [names[i:i + chunk] for i in range(0, len(names), chunk)]
        # 记录进入时全部参数的 requires_grad 快照，probe 结束后恢复
        _rg_snapshot = {n: p.requires_grad for n, p in model.named_parameters()}
        if verbose:
            print(f"[freq_fisher] chunked mode: {len(names)} targets -> {len(groups)} groups "
                  f"of <= {chunk}; {n_batches} batches per group "
                  f"(fwd passes = {len(groups) * n_batches})")
        try:
            for gi, group_names in enumerate(groups):
                group_modules = {n: target_modules[n] for n in group_names}
                # 2026-09-01 修复匹配 bug：target_modules 的 key 是**模块路径**（如
                # "transformer.h.0.attn.q_proj"），而 named_parameters() 返回的是
                # **参数路径**（"transformer.h.0.attn.q_proj.weight"）。此前用
                # `pname in set(模块名)` 精确匹配永远为 False，导致全模型被冻结、
                # loss 无 grad_fn 报 "does not require grad"。这里改为子串匹配：
                # 凡参数路径包含本组任一模块路径（即本组模块及其子参数）即解冻。
                # 非本组参数（含 embedding、LayerNorm、其它层）保持冻结以省梯度显存；
                # 冻结的参数不影响 loss 的 grad_fn（它们以常量参与图，不阻断梯度
                # 流向被解冻的本组权重）。
                group_patterns = tuple(group_names)
                for pname, p in model.named_parameters():
                    p.requires_grad_(any(pat in pname for pat in group_patterns))
                g_counted = 0
                for batch_idx, batch in enumerate(dataloader):
                    if g_counted >= n_batches:
                        break
                    loss, _ = _forward_one_batch(batch)
                    model.zero_grad(set_to_none=True)
                    loss.backward()
                    _accumulate_grad_ffts(group_modules)
                    g_counted += 1
                    if verbose:
                        print(f"  [group {gi+1}/{len(groups)} batch {g_counted}/{n_batches}] "
                              f"loss={float(loss.item()):.4f} elapsed={time.time()-t0:.1f}s")
                counted = max(counted, g_counted)
        finally:
            # 恢复 requires_grad 快照
            for pname, p in model.named_parameters():
                if pname in _rg_snapshot:
                    p.requires_grad_(_rg_snapshot[pname])
        counted = n_batches  # 分组模式下每组都完整消费了 n_batches

    # 4. 求均值
    for k in heatmaps:
        heatmaps[k] /= max(1, counted)

    # 2026-09-01：恢复 gradient checkpointing 状态（若本函数曾启用）
    if _gc_enabled and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        if verbose:
            print("[freq_fisher] gradient checkpointing DISABLED (restored)")

    if verbose:
        print(f"[freq_fisher] 完成 {counted} batches，总耗时 {time.time()-t0:.1f}s，"
              f"覆盖 {len(heatmaps)} 层")
    return heatmaps


# ----------------------------------------------------------------------------
# § 大模型稀疏估计：分块 FFT + fp16
# ----------------------------------------------------------------------------
@torch.enable_grad()
def sparse_freq_fisher_chunked(
    model: nn.Module,
    dataloader: Iterable,
    target_layers: List[str],
    n_batches: int = 4,
    chunk: int = 1024,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """LLaMA-3-8B 等 d=4096 大模型的稀疏估计路径：

    把每层 grad 切成不重叠的 chunk×chunk 块，对每块独立做 2D-FFT 取 |·|²，
    再 placement-back 到原网格。每层只保留一个全分辨率的 |F|² heatmap，
    但中间过程显存恒定为 O(chunk²)。

    复杂度：每层 (d/chunk)² 次 FFT2，每次 O(chunk² log chunk)；
            总计 O(d² log chunk)，相对原来的 O(d² log d) 几乎不省，
            但显存峰值由 O(d²) 降到 O(chunk²)，是大模型的关键。

    注意：分块 FFT 与全图 FFT 不严格等价，块边界处会有少量谱泄漏。
    在元启发式先验场景下完全可接受（先验只需要排序意义上的相对大小）。
    """
    if device is None:
        device = next(model.parameters()).device

    target_modules: Dict[str, nn.Module] = {n: _resolve_module(model, n) for n in target_layers}
    for m in target_modules.values():
        if not isinstance(m, _LINEAR_LIKE):
            raise TypeError(f"target 必须是 nn.Linear 或 transformers Conv1D，实际为 {type(m)}")
        m.weight.requires_grad_(True)

    heatmaps: Dict[str, torch.Tensor] = {
        n: torch.zeros(*m.weight.shape, dtype=torch.float32)
        for n, m in target_modules.items()
    }

    counted = 0
    t0 = time.time()
    for batch in dataloader:
        if counted >= n_batches:
            break
        # 同样用 duck-typing 兼容 BatchEncoding
        if hasattr(batch, "items") and hasattr(batch, "keys"):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            output = model(**batch)
        else:
            batch = tuple(v.to(device) if torch.is_tensor(v) else v for v in batch)
            output = model(*batch)
        loss = _to_loss(output, batch if hasattr(batch, "keys") else {})
        model.zero_grad(set_to_none=True)
        loss.backward()

        for name, m in target_modules.items():
            g = m.weight.grad
            if g is None:
                continue
            # 同样走 CPU + fp32，避开昆仑 aclnnFft / aclnnAbs(complex) 限制
            g_cpu = g.detach().to("cpu", dtype=torch.float32)
            H, W = g_cpu.shape
            heat = heatmaps[name]
            # 分块 FFT2
            for i in range(0, H, chunk):
                for j in range(0, W, chunk):
                    sub = g_cpu[i:i + chunk, j:j + chunk]
                    if sub.numel() == 0:
                        continue
                    Gs = torch.fft.fft2(sub)
                    heat[i:i + Gs.shape[0], j:j + Gs.shape[1]] += (Gs.abs() ** 2)
        counted += 1
        if verbose:
            print(f"  [chunked batch {counted}/{n_batches}] loss={float(loss.item()):.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")

    for k in heatmaps:
        heatmaps[k] /= max(1, counted)
    if verbose:
        print(f"[sparse_freq_fisher_chunked] 完成 {counted} batches，"
              f"总耗时 {time.time()-t0:.1f}s")
    return heatmaps


# ----------------------------------------------------------------------------
# § 由 heatmap 采样 top-k 频点（供搜索算法初始化）
# ----------------------------------------------------------------------------
def sample_topk_indices_from_heatmap(
    heatmap: torch.Tensor, k: int, exclude_dc: bool = True,
    return_2d: bool = True,
) -> torch.Tensor:
    """从频域 heatmap 中选 top-k 频点。

    Args:
        heatmap:    (H, W) 实数张量，值越大表示该频点越"应被建模"
        k:          要选取的频点数（即 FourierFT 的 n_frequency）
        exclude_dc: 是否排除 (0, 0) DC 直流分量（通常在合并到 W₀ 后等价于偏置项，单独建模意义不大）
        return_2d:  True 时返回 [2, k] 的 (row, col) 整数索引（与 fourier_ft.layer 接口一致）；
                    False 时返回 (k,) 的 flat index。

    Returns:
        torch.LongTensor，形状见 return_2d。
    """
    H, W = heatmap.shape
    flat = heatmap.flatten().clone()
    if exclude_dc:
        flat[0] = -float("inf")  # DC 在 fft（未 fftshift）的位置就是 (0, 0)
    if k > flat.numel():
        raise ValueError(f"k={k} 超过总频点数 {flat.numel()}")
    top_vals, top_idx = torch.topk(flat, k=k, largest=True)
    if return_2d:
        rows = (top_idx // W).long()
        cols = (top_idx % W).long()
        return torch.stack([rows, cols], dim=0)
    return top_idx.long()


# ----------------------------------------------------------------------------
# § 一些便捷工具
# ----------------------------------------------------------------------------
def heatmap_to_density(heatmap: torch.Tensor) -> torch.Tensor:
    """归一化为密度图（和为 1），便于做相似度比较或可视化。"""
    s = heatmap.sum().clamp_min(1e-30)
    return heatmap / s


def heatmap_overlap_iou(h1: torch.Tensor, h2: torch.Tensor, k: int) -> float:
    """两个 heatmap 选 top-k 频点的 IoU（用于 RF heatmap vs Fisher heatmap 对比图）。"""
    s1 = set(sample_topk_indices_from_heatmap(h1, k, return_2d=False).tolist())
    s2 = set(sample_topk_indices_from_heatmap(h2, k, return_2d=False).tolist())
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)
