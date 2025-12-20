# Head-Aware Dynamic KV Budgeting

A-Level Paper Project: Efficient Long-Sequence Inference for Large Language Models

## 🎯 Project Goal

Publish an A-level conference/journal paper (ACL, EMNLP, NeurIPS, ICML, etc.)

**Research Question**: How to efficiently compress KV cache for long-sequence inference by leveraging attention head functionality?

**Core Method**: 
- **Head-Aware Cache**: Dynamic KV cache budget allocation based on attention head functionality
- **Group-Aware Eviction**: Collaborative eviction strategy based on head groups

**Baseline Comparisons**:
- H2O (Heavy-Hitter Oracle)
- StreamingLLM (Fixed Window + Attention Sinks)

---

## 🚀 Quick Start

**详细步骤**: 参见 `scripts/ubuntu/README.md`

### Windows系统

```powershell
# 1. 环境准备
python check_environment.py
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 下载模型
.\scripts\windows\download_models.ps1

# 3. 运行实验                                                                                                                                 
.\scripts\windows\run_a_level_experiments.ps1
```

**详细步骤**: 参见 `scripts/windows/README.md`

---

## 📚 Documentation

1. **`scripts/ubuntu/README.md`** ⭐ - Ubuntu系统完整运行指南
2. **`scripts/windows/README.md`** ⭐ - Windows系统完整运行指南
3. **`EXPERIMENT_GUIDE.md`** - 通用实验指南
4. **`A_LEVEL_PAPER_FINAL_GUIDE.md`** - 详细实验流程
5. **`THEORETICAL_ANALYSIS.md`** - 理论分析

---

## 📁 Project Structure

```
StreamingLLM/
├── StreamingLLM_GPE/          # 核心代码（跨平台）
│   ├── baselines/             # Baseline实现 (H2O, StreamingLLM)
│   ├── models/                # 模型实现 (Qwen, Llama, Gemma)
│   ├── evaluate/              # 评估脚本
│   └── utils/                 # 工具函数
│
├── scripts/                    # 系统特定脚本
│   └── windows/               # Windows系统脚本
│       ├── download_models.ps1
│       ├── run_a_level_experiments.ps1
│       └── README.md
│
├── models/                     # 模型文件
├── data_raw/                   # 原始数据
├── output_logs/                # 实验结果输出
│
├── download_models_china.py     # 模型下载（Python 3.9+）
├── download_models_python38.py  # 模型下载（Python 3.8）
│
├── check_environment.py         # 环境检查
├── check_model_integrity.py     # 模型检查
├── test_baselines.py           # Baseline测试
├── analyze_experiment_results.py # 结果分析
├── visualize_results.py        # 可视化
│
└── README.md                    # 本文件
```

---

## 📊 Experiment Requirements

### Must Complete (A-Level Paper)

1. ✅ **Baseline Implementation** (H2O, StreamingLLM)
2. ✅ **Long Sequence Memory Efficiency Comparison**
3. ✅ **Ablation Study** (prove component contributions)
4. ✅ **Multi-Model Validation** (at least 1 model, 3 recommended)

---

## 🔬 Experiments

### Phase 1: Long Sequence Memory Efficiency

- Sequence lengths: 2000, 5000, 10000, 20000 tokens
- Methods: Baseline (GPE), H2O, StreamingLLM, Head-Aware
- Samples: 100 per method

### Phase 2: Budget Impact Analysis

- Budgets: 2048, 4096, 8192 tokens/layer
- Method: Head-Aware

### Phase 3: Ablation Study

- Configurations: Baseline, Head-Aware
- Sequence length: 5000 tokens

### Phase 4: Results Analysis

- Automatic analysis and visualization
- Generate tables and figures for paper

---
