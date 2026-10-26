# ============================================================================
# 文件名：analysis/utils_device.py
# 职责：device 自动识别（NPU/CUDA/CPU 三选一），并提供常用 helper：
#       - get_device(prefer): 返回当前最高优先级可用 device
#       - device_str_for_hf: 返回 'npu'/'cuda'/'cpu' 字符串（HF Trainer 兼容）
#       - empty_cache: 跨平台显存清理
#       - peak_memory_gb: 跨平台峰值显存（GB）
#       - sync_device: 跨平台同步
#       - resolve_npu_id: 从环境变量 ASCEND_RT_VISIBLE_DEVICES 解析当前可见 NPU id
# 输入：可选 prefer 参数（'auto'/'npu'/'cuda'/'cpu'）
# 输出：torch.device 或字符串
# 上游依赖：torch；可选 torch_npu
# 下游消费方：本目录探针脚本
# 版本：teaching-1.0
# 最后更新日期：2026-06-05
# 改编：课程组
# ============================================================================
"""跨设备 helper，统一 NPU / CUDA / CPU 接口。"""
from __future__ import annotations

import os
from typing import Optional

import torch

# 仅在装了 torch_npu 时才会引入 npu 后端；try-except 让 CUDA 环境也能跑
_HAS_NPU = False
try:
    import torch_npu  # noqa: F401
    _HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:
    _HAS_NPU = False


def has_npu() -> bool:
    return _HAS_NPU


def has_cuda() -> bool:
    return torch.cuda.is_available()


def get_device(prefer: str = "auto") -> torch.device:
    """返回当前最高优先级可用 device。

    prefer:
      - "auto": 优先 NPU > CUDA > CPU
      - "npu" / "cuda" / "cpu": 强制（不可用时降级到 CPU 并打印警告）
    """
    prefer = prefer.lower()
    if prefer == "npu" and _HAS_NPU:
        return torch.device("npu")
    if prefer == "cuda" and has_cuda():
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    # auto
    if _HAS_NPU:
        return torch.device("npu")
    if has_cuda():
        return torch.device("cuda")
    return torch.device("cpu")


def device_str() -> str:
    """返回 'npu' / 'cuda' / 'cpu' 字符串（HF Trainer device_map 兼容）。"""
    d = get_device()
    return d.type


def empty_cache():
    """跨平台显存清理。"""
    if _HAS_NPU:
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
    if has_cuda():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def sync_device(device: Optional[torch.device] = None):
    """跨平台 synchronize。"""
    device = device or get_device()
    if device.type == "npu" and _HAS_NPU:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def peak_memory_gb(device: Optional[torch.device] = None) -> float:
    """返回当前 device 峰值显存（GB）；CPU 上返回 0。"""
    device = device or get_device()
    try:
        if device.type == "npu" and _HAS_NPU:
            b = torch.npu.max_memory_allocated()
        elif device.type == "cuda":
            b = torch.cuda.max_memory_allocated()
        else:
            return 0.0
        return float(b) / (1024 ** 3)
    except Exception:
        return 0.0


def reset_peak_memory(device: Optional[torch.device] = None):
    device = device or get_device()
    try:
        if device.type == "npu" and _HAS_NPU:
            torch.npu.reset_peak_memory_stats()
        elif device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def resolve_npu_id() -> int:
    """从 ASCEND_RT_VISIBLE_DEVICES（昇腾）/ CUDA_VISIBLE_DEVICES（NVIDIA）解析当前进程可见的第一张卡 id。"""
    for k in ("ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        v = os.environ.get(k, "")
        if v:
            try:
                return int(v.split(",")[0].strip())
            except Exception:
                pass
    return 0


def describe() -> str:
    """打印当前设备状态（debug 用）。"""
    d = get_device()
    parts = [f"device={d}"]
    try:
        if d.type == "npu" and _HAS_NPU:
            parts.append(f"name={torch.npu.get_device_name(0)}")
            parts.append(f"count={torch.npu.device_count()}")
        elif d.type == "cuda":
            parts.append(f"name={torch.cuda.get_device_name(0)}")
            parts.append(f"count={torch.cuda.device_count()}")
    except Exception:
        pass
    return "  ".join(parts)
