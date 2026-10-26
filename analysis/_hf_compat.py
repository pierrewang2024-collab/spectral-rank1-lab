# ============================================================================
# 文件名：analysis/_hf_compat.py
# 职责：兼容性补丁——在新版 huggingface_hub (0.24+) 与 datasets 2.21 组合下，
#       HfFileSystem.open(..., 'r', encoding='utf-8') 会返回未解压的 gzip
#       二进制流，导致 datasets/load.py:1875 抛 UnicodeDecodeError。
#       本模块的 apply_hf_compat() 在 import 时打 monkey-patch，让旧式 dataset
#       script 路径（e2e_nlg / dart / web_nlg）能正常下载。
# 用法：
#       from _hf_compat import apply_hf_compat
#       apply_hf_compat()         # 必须在 from datasets import ... 之前调用
#       from datasets import load_dataset
# 上游依赖：huggingface_hub
# 下游消费方：analysis/data_loaders.py
# 版本：teaching-1.0（继承 v2.4.2-ds-script-fix 的 HfFileSystem.open 修复）
# 最后更新日期：2026-06-05
# 作者：课程组（源自内部研究代码）
# ============================================================================
"""HF 兼容性 patch：修复 datasets script-based dataset 在新 hub 下的解码 bug。"""
from __future__ import annotations

import gzip
import io


_PATCHED = False


def apply_hf_compat():
    """打 monkey-patch：让 HfFileSystem.open('r') 自动解 gzip 并 decode utf-8。"""
    global _PATCHED
    if _PATCHED:
        return
    try:
        import huggingface_hub
        HF = huggingface_hub.HfFileSystem
    except Exception:
        return
    _orig_open = HF.open

    def _safe_open(self, path, mode="r", **kw):
        # 仅劫持 text 模式；binary 走原路径
        if "b" not in mode and "r" in mode:
            kw_no_encoding = {k: v for k, v in kw.items() if k != "encoding"}
            try:
                with _orig_open(self, path, mode="rb", **kw_no_encoding) as f:
                    data = f.read()
                if data[:2] == b"\x1f\x8b":  # gzip magic
                    data = gzip.decompress(data)
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = data.decode("utf-8", errors="replace")
                return io.StringIO(text)
            except Exception:
                # 兜底：回退到原始接口
                return _orig_open(self, path, mode=mode, **kw)
        return _orig_open(self, path, mode=mode, **kw)

    HF.open = _safe_open
    _PATCHED = True
