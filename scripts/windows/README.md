# Windows实验脚本使用指南

## 📁 脚本结构

```
scripts/windows/
├── config.ps1                    # ⭐ 共享配置文件（统一修改参数）
├── run_all_experiments.ps1      # ⭐ 主脚本：一键运行所有实验
│
├── phase0_oracle_batch.ps1      # Phase 0: Oracle Batch验证（可选）
├── phase1_long_sequence.ps1     # Phase 1: 长序列内存效率对比（必须）
├── phase2_budget_analysis.ps1   # Phase 2: 预算影响分析（必须）
├── phase3_ablation.ps1          # Phase 3: 消融实验（必须）
├── phase4_analysis.ps1           # Phase 4: 结果分析和可视化（推荐）
├── phase5_haq_kv.ps1            # Phase 5: HAQ-KV实验（可选）
│
├── download_models.ps1           # 模型下载脚本
├── run_head_aware_experiments.ps1  # Head-Aware快速测试脚本
└── README.md                     # 本文件
```

## 🚀 快速开始

### 方式1: 一键运行所有实验（推荐）

```powershell
# 运行所有Phase的实验（Phase 0-4）
.\scripts\windows\run_all_experiments.ps1
```

### 方式2: 分阶段运行

```powershell
# Phase 0: Oracle Batch验证（可选，确定性能上限）
.\scripts\windows\phase0_oracle_batch.ps1

# Phase 1: 长序列对比（必须）
.\scripts\windows\phase1_long_sequence.ps1

# Phase 2: 预算分析（必须）
.\scripts\windows\phase2_budget_analysis.ps1

# Phase 3: 消融实验（必须）
.\scripts\windows\phase3_ablation.ps1

# Phase 4: 结果分析（推荐）
.\scripts\windows\phase4_analysis.ps1

# Phase 5: HAQ-KV实验（可选）
.\scripts\windows\phase5_haq_kv.ps1
```

## ⚙️ 配置修改

**所有脚本共享 `config.ps1` 配置文件**，修改此文件即可统一调整参数：

```powershell
# 编辑配置文件
notepad .\scripts\windows\config.ps1

# 主要参数：
# - $MAX_SAMPLES = 100        # 样本数量（完整实验）
# - $TOTAL_BUDGET = 2048     # KV cache预算
# - $LONG_SEQUENCE_LENGTHS   # 测试的序列长度
# - $BUDGETS                 # 预算分析的值
```

## 🎯 运行选项

### 运行所有实验

```powershell
.\scripts\windows\run_all_experiments.ps1
```

### 跳过某些Phase

```powershell
# 跳过Phase 0（Oracle Batch验证）
.\scripts\windows\run_all_experiments.ps1 -SkipPhase0

# 跳过Phase 4（结果分析，只运行实验）
.\scripts\windows\run_all_experiments.ps1 -SkipPhase4

# 同时跳过多个
.\scripts\windows\run_all_experiments.ps1 -SkipPhase0 -SkipPhase4
```

### 只运行特定Phase

```powershell
# 只运行Phase 1
.\scripts\windows\run_all_experiments.ps1 -Phase1Only

# 只运行Phase 2
.\scripts\windows\run_all_experiments.ps1 -Phase2Only

# 只运行Phase 4（分析已有结果）
.\scripts\windows\run_all_experiments.ps1 -Phase4Only
```

## 📊 实验说明

### Phase 0: Oracle Batch验证（可选）
- **目的**: 确定模型在batch模式下的性能上限（Upper Bound）
- **时间**: ~10分钟
- **输出**: `output_logs/a_level_paper/oracle_batch/`
- **用途**: 作为性能参考，帮助理解streaming vs batch的差异

### Phase 1: 长序列内存效率对比（必须）⭐⭐⭐
- **对比方法**: 
  1. Baseline (GPE)
  2. H2O Baseline
  3. StreamingLLM Baseline
  4. Head-Aware ⭐（核心创新）
- **序列长度**: 2000, 5000, 10000, 20000 tokens
- **时间**: ~4-8小时（取决于硬件）
- **输出**: `output_logs/a_level_paper/long_seq_{长度}/{方法}/`
- **重要性**: 这是核心对比实验，证明Head-Aware的优势

### Phase 2: 预算影响分析（必须）⭐⭐
- **方法**: Head-Aware
- **预算**: 2048, 4096, 8192 tokens/layer
- **时间**: ~1-2小时
- **输出**: `output_logs/a_level_paper/budget_{预算}/`
- **用途**: 分析不同预算对性能的影响

### Phase 3: 消融实验（必须）⭐⭐
- **对比**: 
  1. Baseline (GPE only)
  2. Head-Aware ⭐
- **序列长度**: 5000 tokens
- **时间**: ~1-2小时
- **输出**: `output_logs/a_level_paper/ablation/{方法}/`
- **用途**: 证明Head-Aware的有效性

### Phase 4: 结果分析和可视化（推荐）⭐
- **功能**: 分析结果，生成表格和图表
- **时间**: ~10-30分钟
- **输出**: 
  - CSV/JSON/LaTeX表格（用于论文）
  - 可视化图表（PNG/PDF）
- **输出位置**: `output_logs/a_level_paper/*_summary.*` 和 `figures/`

### Phase 5: HAQ-KV实验（可选）
- **方法**: HAQ-KV（Head-Aware Quantized KV Cache）
- **功能**: 测试异构量化方法
- **时间**: ~2-4小时
- **输出**: `output_logs/a_level_paper/haq_kv_{bits}bit/`
- **用途**: 验证量化方法的有效性

## 📝 实验流程示例

### 完整流程（推荐）

```powershell
# 1. 下载模型（如果还没有）
.\scripts\windows\download_models.ps1

# 2. 一键运行所有实验
.\scripts\windows\run_all_experiments.ps1

# 3. 等待完成（可能需要数小时到数天）
```

### 分步运行（适合调试）

```powershell
# 1. 先运行Phase 1验证（小样本）
# 修改config.ps1: $MAX_SAMPLES = 10
.\scripts\windows\phase1_long_sequence.ps1

# 2. 检查结果
# 3. 如果结果好，修改回100样本，继续运行其他Phase
.\scripts\windows\phase2_budget_analysis.ps1
.\scripts\windows\phase3_ablation.ps1
.\scripts\windows\phase4_analysis.ps1
```

### 快速测试（小样本）

```powershell
# 修改config.ps1中的$MAX_SAMPLES = 10
# 然后运行
.\scripts\windows\run_all_experiments.ps1 -SkipPhase0
```

## 🔧 自定义参数

### 方法1: 修改config.ps1（推荐）

编辑 `scripts/windows/config.ps1`，修改参数后所有脚本都会使用新参数。

**主要参数**:
- `$MAX_SAMPLES = 100` - 样本数量（完整实验用100，测试用10）
- `$TOTAL_BUDGET = 2048` - KV cache预算
- `$LONG_SEQUENCE_LENGTHS = @(2000, 5000, 10000, 20000)` - 序列长度
- `$BUDGETS = @(2048, 4096, 8192)` - 预算分析值

### 方法2: 运行时传递参数（部分脚本支持）

```powershell
# Phase 1: 自定义序列长度和样本数
.\scripts\windows\phase1_long_sequence.ps1 `
    -SequenceLengths @(2000, 5000) `
    -Samples 50

# Phase 2: 自定义预算值
.\scripts\windows\phase2_budget_analysis.ps1 `
    -Budgets @(1024, 2048, 4096) `
    -Samples 50

# Phase 5: 自定义量化位数
.\scripts\windows\phase5_haq_kv.ps1 -RetrievalBits 2
```

## ⚠️ 注意事项

1. **执行策略**: 如果遇到执行策略错误，运行：
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **路径**: 确保在项目根目录运行脚本

3. **模型**: 运行前确保模型已下载

4. **显存**: 如果显存不足，可以：
   - 减少 `$MAX_SAMPLES`（在config.ps1中）
   - 减少序列长度数量
   - 使用量化（需要修改脚本添加 `--quantization 4bit`）

5. **中断恢复**: 如果实验中断，可以：
   - 只运行未完成的Phase
   - 使用 `-Phase4Only` 分析已有结果

## 📈 预期时间

- **Phase 0**: ~10分钟
- **Phase 1**: ~4-8小时（4个序列长度 × 4种方法 × 100样本）
- **Phase 2**: ~1-2小时（3个预算 × 100样本）
- **Phase 3**: ~1-2小时（2种配置 × 100样本）
- **Phase 4**: ~10-30分钟
- **Phase 5**: ~2-4小时（HAQ-KV实验）

**总计**: 约6-12小时（取决于硬件配置）

## 🎯 输出文件结构

```
output_logs/a_level_paper/
├── oracle_batch/              # Phase 0结果
│   └── results.json
│
├── long_seq_2000/             # Phase 1结果
│   ├── baseline/
│   ├── h2o/
│   ├── streamingllm/
│   └── head_aware/
│
├── long_seq_5000/
├── long_seq_10000/
├── long_seq_20000/
│
├── budget_2048/               # Phase 2结果
├── budget_4096/
├── budget_8192/
│
├── ablation/                  # Phase 3结果
│   ├── baseline/
│   └── head_aware/
│
├── haq_kv_4bit/               # Phase 5结果（如果运行）
│   └── long_seq_*/
│
├── long_seq_10000_summary.csv # Phase 4分析结果
├── ablation_summary.csv
├── long_seq_10000_table.tex   # LaTeX表格（用于论文）
└── figures/                   # 可视化图表
    ├── bleu_comparison.png
    ├── memory_comparison.png
    └── ...
```

## 🆘 故障排除

### 问题1: 脚本无法执行

```powershell
# 检查执行策略
Get-ExecutionPolicy

# 设置执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题2: 模型未找到

```powershell
# 下载模型
.\scripts\windows\download_models.ps1
```

### 问题3: 显存不足

- 减少 `$MAX_SAMPLES`（在config.ps1中）
- 减少序列长度数量
- 使用量化（需要修改脚本添加 `--quantization 4bit`）

### 问题4: 只想重新运行某个Phase

```powershell
# 只运行Phase 1
.\scripts\windows\run_all_experiments.ps1 -Phase1Only

# 只分析已有结果
.\scripts\windows\run_all_experiments.ps1 -Phase4Only
```

## 📚 相关文档

- `A_LEVEL_PAPER_FINAL_GUIDE.md` - 完整实验指南
- `HAQ_KV_README.md` - HAQ-KV方法说明
- `GROUP_AWARE_REMOVAL_SUMMARY.md` - Group-Aware移除说明

## 🎓 实验设计说明

### 核心创新点

**Head-Aware Dynamic KV Budgeting** 是唯一的核心创新点：

1. **Head功能分析**: 根据attention patterns将heads分为Retrieval/Induction/Local三类
2. **异构预算分配**: 不同head类型分配不同的KV cache预算
3. **异构量化** (HAQ-KV): Retrieval Heads使用低精度量化，存储更多历史

### 对比方法

- **Baseline (GPE)**: 原始方法，无压缩
- **H2O**: 统一压缩方法（SOTA baseline）
- **StreamingLLM**: 滑动窗口方法（SOTA baseline）
- **Head-Aware**: 我们的方法（核心创新）

### 实验目标

1. 证明Head-Aware在内存效率上的优势
2. 证明Head-Aware在性能保持上的优势
3. 分析不同预算的影响
4. 通过消融实验证明方法有效性
