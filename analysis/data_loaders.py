# ============================================================================
# 文件名：analysis/data_loaders.py
# 职责：统一加载 NLG（E2E/WebNLG/DART）+ NLU（GLUE: SST-2/MNLI/QNLI/RTE）数据集，
#       返回 HF Dataset 对象 + tokenize 后的训练 / 验证 / 测试集。
# 输入：task_cfg dict（含 dataset/name/kind/max_length/num_labels），tokenizer
# 输出：(train_ds, eval_ds, test_ds, raw_eval_ds)；NLG 时 raw_eval_ds 含原始
#       reference 用于 BLEU 计算。
# 上游依赖：datasets、transformers
# 下游消费方：本目录探针脚本
# 版本：teaching-1.0
# 最后更新日期：2026-06-05
# 改编：课程组
# ============================================================================
"""HF datasets + tokenize 统一加载入口（NLG / NLU）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# datasets 2.21 + huggingface_hub 0.24+ 组合下脚本式数据集（e2e_nlg/dart/web_nlg）
# 会触发 HfFileSystem.open 解码 bug；这里在 import datasets 前打 monkey-patch。
try:
    from _hf_compat import apply_hf_compat
    apply_hf_compat()
except Exception:
    pass


# ---------------------- public API ----------------------

def load_task(task_cfg: Dict[str, Any], tokenizer, *,
              cache_dir: Optional[str] = None,
              max_train_samples: Optional[int] = None,
              max_eval_samples: Optional[int] = None):
    """统一入口。

    返回 (train_ds, eval_ds, test_ds, raw_eval_ds, data_collator, info)
      - train_ds / eval_ds / test_ds: 已 tokenize 的 HF Dataset
      - raw_eval_ds: 仅 NLG 用——保留原始 input / references 以便 model.generate 后算 BLEU
      - data_collator: 配套的 DataCollator（CausalLM / SeqCls）
      - info: dict，含 num_labels / label_list / kind 等元信息
    """
    kind = task_cfg.get("kind", "nlg")
    if kind == "nlg":
        return _load_nlg(task_cfg, tokenizer,
                         cache_dir=cache_dir,
                         max_train_samples=max_train_samples,
                         max_eval_samples=max_eval_samples)
    elif kind == "nlu":
        return _load_nlu(task_cfg, tokenizer,
                         cache_dir=cache_dir,
                         max_train_samples=max_train_samples,
                         max_eval_samples=max_eval_samples)
    else:
        raise ValueError(f"未知 task.kind: {kind}")


# ---------------------- NLG ----------------------

# NLG 任务的"输入字段 → 文本拼接"规则
# 数据集 schema 在不同 HF 版本里可能略有差异，这里写得尽量鲁棒
_NLG_TEMPLATES = {
    "e2e_nlg":  {"src_field": "meaning_representation", "tgt_field": "human_reference"},
    "web_nlg":  {"src_field": None, "tgt_field": None},  # 见 _format_webnlg
    "dart":     {"src_field": None, "tgt_field": None},  # 见 _format_dart
}


def _format_webnlg(example):
    """WebNLG 的 modified_triple_sets / lex 字段。"""
    triples = example.get("modified_triple_sets", {}).get("mtriple_set", [[]])
    if isinstance(triples, list) and triples and isinstance(triples[0], list):
        triples = triples[0]
    src = " | ".join(str(t) for t in triples) if triples else ""
    lex = example.get("lex", {})
    refs = lex.get("text", []) if isinstance(lex, dict) else []
    tgt = refs[0] if refs else ""
    return src, tgt, refs


def _format_dart(example):
    """DART 的 tripleset / annotations.text 字段。"""
    tripleset = example.get("tripleset", [])
    src = " | ".join([" : ".join(t) for t in tripleset]) if tripleset else ""
    ann = example.get("annotations", {})
    refs = ann.get("text", []) if isinstance(ann, dict) else []
    tgt = refs[0] if refs else ""
    return src, tgt, refs


def _format_e2e(example):
    src = example.get("meaning_representation", "")
    tgt = example.get("human_reference", "")
    refs = [tgt]
    return src, tgt, refs


def _format_alpaca(example):
    """Alpaca-Cleaned。字段：instruction / input / output。

    NLG eval 框架期望返回 (src, tgt, refs)：
      src = "<instruction>\\n\\n<input>"（input 为空则只留 instruction）
      tgt = output；refs = [output]（alpaca-eval 不提供多 ref）
    """
    instr = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()
    src = instr if not inp else f"{instr}\n\n{inp}"
    return src, out, [out]


def _format_samsum(example):
    """SAMSum 对话摘要。字段：dialogue / summary。单参考。"""
    src = (example.get("dialogue") or "").strip()
    tgt = (example.get("summary") or "").strip()
    return src, tgt, [tgt]


_NLG_FORMATTERS = {
    "e2e_nlg": _format_e2e,
    "web_nlg": _format_webnlg,
    "dart":    _format_dart,
    "samsum":  _format_samsum,
}


def _resolve_nlg_dataset(name: str):
    """name -> (hf_path, hf_config, formatter, eval_split)"""
    name = name.lower()
    if name in ("e2e", "e2e_nlg"):
        return ("e2e_nlg", None, _format_e2e, "validation")
    if name in ("webnlg", "web_nlg"):
        return ("web_nlg", "release_v3.0_en", _format_webnlg, "dev")
    if name in ("dart",):
        return ("dart", None, _format_dart, "validation")
    if name in ("samsum",):
        # 原作者 2024 年将 samsum 从 Hub 下架；依次尝试官方迁移名与 parquet 镜像。
        # _load_nlg 会对候选列表逐个尝试，任一命中即用（schema 相同：dialogue/summary）。
        return (["samsum", "Samsung/samsum", "knkarthick/samsum", "gimmaru/samsum"],
                None, _format_samsum, "validation")
    if name in ("alpaca", "alpaca_cleaned", "yahma/alpaca-cleaned"):
        # 需提前用 scripts/prepare_alpaca_data.py 缓存到 .hf_cache/datasets
        return ("yahma/alpaca-cleaned", None, _format_alpaca, "train")
    raise ValueError(f"未知 NLG dataset: {name}")


def _load_nlg(task_cfg, tokenizer, *, cache_dir, max_train_samples, max_eval_samples):
    from datasets import load_dataset
    from transformers import default_data_collator

    ds_name = task_cfg.get("dataset", task_cfg.get("name", "e2e_nlg"))
    hf_path, hf_cfg, formatter, eval_split = _resolve_nlg_dataset(ds_name)
    max_length = int(task_cfg.get("max_length", 128))

    # hf_path 支持候选列表（如 samsum 原仓库下架后的多个镜像），逐个尝试
    candidates = hf_path if isinstance(hf_path, (list, tuple)) else [hf_path]
    ds = None
    last_err = None
    for cand in candidates:
        try:
            if hf_cfg:
                ds = load_dataset(cand, hf_cfg, cache_dir=cache_dir, trust_remote_code=True)
            else:
                ds = load_dataset(cand, cache_dir=cache_dir, trust_remote_code=True)
            print(f"  [data] loaded dataset from: {cand}")
            break
        except Exception as e:
            last_err = e
            print(f"  [data] candidate failed: {cand} ({type(e).__name__}); trying next...")
    if ds is None:
        raise RuntimeError(f"所有候选数据源均失败: {candidates}; last error: {last_err}")

    train_split = "train" if "train" in ds else list(ds.keys())[0]
    # 单 split 数据集（如 alpaca 只有 "train"）自动切 90/5/5（train/val/test）
    if len(ds.keys()) == 1:
        full = ds[train_split]
        # seed 固定保证可复现，shuffle 后切分
        full = full.shuffle(seed=42)
        n = len(full)
        n_train = int(n * 0.90)
        n_val = int(n * 0.05)
        ds = {
            "train":      full.select(range(0, n_train)),
            "validation": full.select(range(n_train, n_train + n_val)),
            "test":       full.select(range(n_train + n_val, n)),
        }
        train_split, eval_split, test_split = "train", "validation", "test"
        print(f"  [data] single-split dataset auto-split → train={len(ds['train'])} "
              f"val={len(ds['validation'])} test={len(ds['test'])}")
    else:
        if eval_split not in ds:
            eval_split = "validation" if "validation" in ds else \
                         "valid" if "valid" in ds else \
                         "test" if "test" in ds else list(ds.keys())[-1]
        test_split = "test" if "test" in ds else eval_split

    def _to_text(example):
        src, tgt, refs = formatter(example)
        prompt = f"input: {src}\noutput: "
        full   = prompt + str(tgt) + tokenizer.eos_token
        return {"prompt": prompt, "full_text": full, "target": str(tgt), "references": refs}

    raw_train = ds[train_split].map(_to_text, remove_columns=ds[train_split].column_names)
    raw_eval  = ds[eval_split].map(_to_text,  remove_columns=ds[eval_split].column_names)
    raw_test  = ds[test_split].map(_to_text,  remove_columns=ds[test_split].column_names)

    if max_train_samples:
        raw_train = raw_train.select(range(min(max_train_samples, len(raw_train))))
    if max_eval_samples:
        raw_eval  = raw_eval.select(range(min(max_eval_samples,  len(raw_eval))))
        raw_test  = raw_test.select(range(min(max_eval_samples,  len(raw_test))))

    def _tokenize_train(ex):
        # 全文 tokenize，prompt 部分 label = -100 不参与 loss
        enc = tokenizer(ex["full_text"], truncation=True, max_length=max_length,
                        padding="max_length", return_tensors=None)
        input_ids = enc["input_ids"]
        labels = list(input_ids)
        # 计算 prompt 长度，把 prompt 部分的 label 置 -100
        prompt_ids = tokenizer(ex["prompt"], truncation=True, max_length=max_length,
                               return_tensors=None)["input_ids"]
        plen = min(len(prompt_ids), len(labels))
        labels[:plen] = [-100] * plen
        # pad 部分也 -100
        pad_id = tokenizer.pad_token_id
        labels = [-100 if t == pad_id else l for t, l in zip(input_ids, labels)]
        enc["labels"] = labels
        return enc

    def _tokenize_eval(ex):
        # 评测时只 tokenize prompt（用 generate 续写）
        enc = tokenizer(ex["prompt"], truncation=True, max_length=max_length,
                        padding="max_length", return_tensors=None)
        return enc

    train_ds = raw_train.map(_tokenize_train,
                             remove_columns=[c for c in raw_train.column_names if c != "references"])
    eval_ds  = raw_eval.map(_tokenize_eval,
                            remove_columns=[c for c in raw_eval.column_names
                                            if c not in ("references", "target", "prompt")])
    test_ds  = raw_test.map(_tokenize_eval,
                            remove_columns=[c for c in raw_test.column_names
                                            if c not in ("references", "target", "prompt")])

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 关键：必须用 default_data_collator 而不是 DataCollatorForLanguageModeling(mlm=False)。
    # 后者会无条件地把 batch["labels"] 覆盖成 input_ids.clone()（仅把 pad 位置置 -100），
    # 这会丢掉我们在 _tokenize_train 中精心设置的"prompt 段 labels=-100"——结果模型把 prompt
    # 也当训练目标，对 LoRA 这类大容量 adapter 影响有限，但对 0.048M 的 FourierFT 直接致命：
    # 训练目标里塞满 "input: ... output: " 的复读，几百步后生成全是 *** / --- 等 token。
    collator = default_data_collator
    info = {"kind": "nlg", "dataset": ds_name}
    return train_ds, eval_ds, test_ds, raw_eval, collator, info


# ---------------------- NLU ----------------------

_GLUE_KEYS = {
    "sst2":  ("sentence", None,        2, "validation"),
    "mnli":  ("premise",  "hypothesis", 3, "validation_matched"),
    "qnli":  ("question", "sentence",  2, "validation"),
    "rte":   ("sentence1", "sentence2", 2, "validation"),
}


def _load_nlu(task_cfg, tokenizer, *, cache_dir, max_train_samples, max_eval_samples):
    from datasets import load_dataset
    from transformers import DataCollatorWithPadding

    name = task_cfg.get("name", "sst2").lower()
    if "/" in str(task_cfg.get("dataset", "")):
        # config 'glue/rte' -> ('glue', 'rte')
        sub = task_cfg["dataset"].split("/")[-1]
    else:
        sub = name

    if sub not in _GLUE_KEYS:
        raise ValueError(f"未知 NLU 任务: {sub}")
    s1, s2, num_labels, eval_split = _GLUE_KEYS[sub]
    max_length = int(task_cfg.get("max_length", 128))

    ds = load_dataset("glue", sub, cache_dir=cache_dir, trust_remote_code=True)
    train_split = "train" if "train" in ds else list(ds.keys())[0]
    if eval_split not in ds:
        eval_split = "validation" if "validation" in ds else list(ds.keys())[-1]

    def _tok(ex):
        if s2:
            return tokenizer(ex[s1], ex[s2], truncation=True, max_length=max_length)
        return tokenizer(ex[s1], truncation=True, max_length=max_length)

    train_ds = ds[train_split].map(_tok, batched=True)
    eval_ds  = ds[eval_split].map(_tok,  batched=True)
    test_ds  = eval_ds  # GLUE 的 test split label 不公开，eval = test

    keep = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in train_ds.column_names: keep.append("token_type_ids")
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    eval_ds  = eval_ds.remove_columns([c for c in eval_ds.column_names  if c not in keep])
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds  = eval_ds.rename_column("label", "labels")

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples:
        eval_ds  = eval_ds.select(range(min(max_eval_samples,  len(eval_ds))))
    test_ds = eval_ds

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    info = {"kind": "nlu", "dataset": sub, "num_labels": num_labels}
    return train_ds, eval_ds, test_ds, eval_ds, collator, info
