# analysis/ — 频域微调审计：Fisher 选点探针与谱分析工具

本目录是"频域微调：频点选择与谱结构分析"课程实验的工具包，改编自课程组内部研究代码。基础方法 FourierFT 见 Gao et al. 2024（arXiv:2405.03003）。

## 文件说明

| 文件 | 作用 | 依赖 |
|------|------|------|
| `freq_fisher_heatmap.py` | 核心：Empirical Fisher 谱计算。对每层目标权重取 ∇W，做 2D-DFT，在 batch 上平均 \|·\|² | torch（transformers 可选） |
| `export_fisher_probe_indices.py` | 探针驱动脚本：加载 GPT-2-Medium + WebNLG，调用上面的核心，导出 top-K 频点字典 `fisher_indices.pt` | torch、transformers、datasets |
| `spectral_energy_analysis.py` | 训后谱分析：读 FourierFT checkpoint，重建 ΔW，算有效秩/谱熵/共轭成对/行列集中度 | **仅 numpy**（无 torch 依赖） |
| `data_loaders.py` / `utils_device.py` / `_hf_compat.py` | 上两个脚本的支撑模块（数据加载/设备检测/HF 兼容补丁） | datasets、transformers |

另附：
- `results/`：两个**参考 checkpoint**（GPT-2-Medium WebNLG seed42，Fisher 与 Vanilla 各 ~1.9MB），目录名匹配脚本的 glob 模式，克隆后即可直接运行谱分析。
- `spectral_energy_reference.json`：完整参考结果（完整参考谱统计数字），供复现对照。

## 快速开始 1：谱分析（复现参考数字，纯 numpy，笔记本 CPU 即可）

```bash
# 开箱即用：仓库自带参考 checkpoint，默认 --root 即仓库根
python3 analysis/spectral_energy_analysis.py          # 全量：GPT-2 全 24 层，约 1–2 分钟
python3 analysis/spectral_energy_analysis.py --smoke  # 自检：只算第 0 层，秒级
# 分析自己训练得到的 checkpoint：--root 指向含 results/ 的目录，目录名需匹配内置 glob，如
#   results/FISHER96K_B64_myrun__evalfix_main_fish96k_s42/final_adapter.pt
# （没有 Llama checkpoint 时对应行显示 MISS 并自动跳过，属正常）
```

输出 `analysis/spectral_energy.json`。**参考答案（GPT-2-Medium，seed 42，每层 1000 频点）**：

| 模块类型 | 选点 | 有效秩 | distinct R/Q | 共轭成对比例 |
|----------|------|--------|--------------|--------------|
| attn.c_attn | Fisher | 38.0 | 143 / 126 | 1.000 |
| attn.c_proj | Fisher | 38.5 | 136 / 115 | 1.000 |
| mlp.c_fc | Fisher | 40.2 | 234 / 150 | 1.000 |
| **mlp.c_proj** | **Fisher** | **1.01** | **1000 / 1** | **0.998** |
| 各模块 | Vanilla（随机） | 578–736 | ~650–890 / ~640–890 | 0.000 |

关键现象：Fisher 选点在 `mlp.c_proj` 的 **全部 24 层中有 23 层** 把 1000 个频点全部选在**同一条谱列**（即 in 维度的第 0 个频率，DC 列），使 ΔW 精确秩 1；其余三类投影的有效秩也比随机选点低一个数量级。

## 快速开始 2：重新生成 Fisher 热图/频点（需要 torch，联网下载模型与数据）

```bash
pip install torch transformers datasets
# GPT-2-Medium（~1.4GB）与 WebNLG 首次运行自动下载
python3 analysis/export_fisher_probe_indices.py \
    --n_frequency 1000 --output_dir out_fisher_demo
```

输出：`out_fisher_demo/fisher_indices.pt`（96 层 × [2,1000] 频点）与可审计的元数据 json。默认协议 = 训练集前 64 样本（8 batch × 8），与参考协议一致；换窗口用 `--probe_mode offset --start_batch 8`。

## 已知陷阱（来自主仓库开发史，学生必读）

1. **FFT 网格尺寸**：重建 ΔW 必须按真实层尺寸 `(out_features, in_features)` 建 dense 谱矩阵。若用 `indices.max()+1` 推断网格，`mlp.c_proj` 的选点全在第 0 列时会被错推断成 1024×1，得到错误的"秩 1"（本脚本 v2 曾踩坑，v3 已修）。
2. **Conv1D 转置**：HuggingFace GPT-2 的投影层是 `Conv1D`，权重存储为 `[in, out]`，与 `nn.Linear` 相反。谱统计前必须转成 `[out, in]`，否则行列集中度对调。
3. **实部 DFT 的共轭对称**：谱位置 q 与 -q 对应同一个余弦原子；Fisher 分数 |\ FFT2(∇W)\|² 天然共轭对称，所以 top-K 选点成对出现，K 个系数只张成约 K/2 个独立原子。

## 引用

- 基础方法 FourierFT：Gao et al., *FourierFT: Fine-Tuning in the Spectral Frequency Domain*, 2024（arXiv:2405.03003）。
- 探针协议（GPT-2-Medium / WebNLG / FP32 / 64 样本 / batch 8）与参考 checkpoint 的训练协议一致。
