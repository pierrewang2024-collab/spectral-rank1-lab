# spectral-rank1-lab：GPT-2 权重更新的频域指纹（课程实验材料）

本仓库配套大四实验课课题《GPT-2 权重更新的频域指纹：一次梯度回传为什么会选出秩 1 更新？》。

微调语言模型时，频域方法 **FourierFT** 把权重更新 ΔW 表达为少数二维傅里叶系数，训练前用一次梯度回传给每个频点打分（Empirical Fisher 谱）选 top-K。已知观测：在 GPT-2-Medium 上，Fisher 选出的 1000 个频点在 `mlp.c_proj` 的 24 层中有 23 层**全部落在同一条谱列**，使 ΔW 精确秩 1——一次"频域"选点隐式地重新发明了 LoRA。为什么恰好是这一列？换模型还成立吗？本课题的任务就是回答这些问题。

**实验任务、必做/选做要求与验收标准见课程指导书（LMS 发放）。本 README 只覆盖环境与工具用法。**

## 目录结构

```
├── analysis/                          # 工具包（用法详见 analysis/README.md）
│   ├── spectral_energy_analysis.py    #   谱分析：读 checkpoint → 有效秩/谱熵/共轭成对（纯 numpy）
│   ├── freq_fisher_heatmap.py         #   Fisher 探针核心：∇W → 2D FFT → 能量谱（torch）
│   ├── export_fisher_probe_indices.py #   探针驱动脚本：GPT-2-Medium + WebNLG → top-K 频点
│   ├── data_loaders.py / utils_device.py / _hf_compat.py   # 支撑模块
│   └── spectral_energy_reference.json # 参考谱统计（对照用）
└── results/                           # 两个参考 checkpoint（GPT-2-Medium seed42，各 1.9MB）
```

## 快速开始

**任务一（纯 numpy，任何笔记本，分钟级）**——仓库自带参考 checkpoint，开箱即用：

```bash
pip install numpy
python3 analysis/spectral_energy_analysis.py --smoke   # 秒级自检
python3 analysis/spectral_energy_analysis.py           # 全量：GPT-2 全 24 层，约 1–2 分钟
```

预期首屏输出包含：`gpt2 Fisher mlp.c_proj eff_rank=1.00 ... R=1000 Q=1`。

**任务二（需 torch，联网下载模型与数据，CPU 约 0.5–1 小时）**：

```bash
pip install -r requirements.txt
python3 analysis/export_fisher_probe_indices.py --n_frequency 1000 --output_dir my_probe
```

首次运行自动下载 GPT-2-Medium（~1.4GB）与 WebNLG。输出 `my_probe/fisher_indices.pt`（96 层 × [2,1000] 频点）。

## 环境要求

- Python 3.10+；任务一只需 numpy，任务二见 `requirements.txt`
- 磁盘：克隆仓库 ~10MB；任务二另需 ~2GB（模型 + 数据）
- GPU：不强制（CPU 可完成；有 GPU 更快）
- 内存：任务二建议 16GB（8GB 机器把 `--probe_batch_size` 降到 4 并在报告注明）

## 已知陷阱（开工前必读）

见 `analysis/README.md` 末节：FFT 网格尺寸推断、Conv1D 权重转置、实部 DFT 的共轭对称。

## 引用

- 基础方法 FourierFT：Gao et al., *FourierFT: Fine-Tuning in the Spectral Frequency Domain*, 2024（arXiv:2405.03003）
- 本仓库工具改编自课程组内部研究代码，仅用于教学。
