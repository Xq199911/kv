# 实验脚本总览

## 📁 脚本组织结构

```
scripts/
├── README.md                    # 本文件（脚本总览）
│
├── windows/                     # Windows系统脚本
│   ├── README.md               # Windows脚本详细说明
│   ├── config.ps1              # ⭐ 共享配置文件
│   ├── run_all_experiments.ps1 # ⭐ 一键运行所有实验
│   │
│   ├── phase0_oracle_batch.ps1      # Phase 0: Oracle Batch验证
│   ├── phase1_long_sequence.ps1     # Phase 1: 长序列对比
│   ├── phase2_budget_analysis.ps1   # Phase 2: 预算分析
│   ├── phase3_ablation.ps1          # Phase 3: 消融实验
│   ├── phase4_analysis.ps1          # Phase 4: 结果分析
│   ├── phase5_haq_kv.ps1            # Phase 5: HAQ-KV实验
│   │
│   ├── download_models.ps1         # 模型下载
│   └── run_head_aware_experiments.ps1  # Head-Aware快速测试
│
└── run_head_aware_experiments.py    # Python版本（跨平台）
```

## 🚀 快速开始

### Windows用户

```powershell
# 1. 下载模型（如果还没有）
.\scripts\windows\download_models.ps1

# 2. 一键运行所有实验
.\scripts\windows\run_all_experiments.ps1
```

### 详细说明

- **Windows用户**: 查看 `scripts/windows/README.md`
- **配置修改**: 编辑 `scripts/windows/config.ps1`

## 📊 实验Phase说明

### Phase 0: Oracle Batch验证（可选）
- 确定batch模式下的性能上限
- 时间: ~10分钟

### Phase 1: 长序列内存效率对比（必须）⭐⭐⭐
- 对比4种方法：Baseline, H2O, StreamingLLM, Head-Aware
- 测试4个序列长度：2000, 5000, 10000, 20000 tokens
- 时间: ~4-8小时

### Phase 2: 预算影响分析（必须）⭐⭐
- 测试Head-Aware在不同预算下的表现
- 预算：2048, 4096, 8192 tokens/layer
- 时间: ~1-2小时

### Phase 3: 消融实验（必须）⭐⭐
- 对比Baseline vs Head-Aware
- 时间: ~1-2小时

### Phase 4: 结果分析和可视化（推荐）⭐
- 生成表格和图表
- 时间: ~10-30分钟

### Phase 5: HAQ-KV实验（可选）
- 测试异构量化方法
- 时间: ~2-4小时

## 🎯 核心创新点

**Head-Aware Dynamic KV Budgeting** 是唯一的核心创新点：

1. Head功能分析（Retrieval/Induction/Local）
2. 异构预算分配
3. 异构量化（HAQ-KV）

## 📚 相关文档

- `A_LEVEL_PAPER_FINAL_GUIDE.md` - 完整实验指南
- `HAQ_KV_README.md` - HAQ-KV方法说明
- `scripts/windows/README.md` - Windows脚本详细说明

