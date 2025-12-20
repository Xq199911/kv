# A级论文完整实验指南（最终版）

## 🎯 目标
发表A级会议/期刊论文（ACL, EMNLP, NeurIPS, ICML等）

---

## 📋 完整执行流程（按顺序）

### Step 0: 环境准备

**操作步骤**:

1. **检查Python环境**:
```bash
python --version  # 需要Python 3.8+
```

2. **检查CUDA环境** (如果使用GPU):
```bash
nvidia-smi  # 检查GPU是否可用
```

3. **检查环境依赖**:
```bash
python check_environment.py
```

4. **安装Python依赖**:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

5. **安装ModelScope** (用于国内下载模型):
```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**预期输出**: 所有检查通过，无错误信息

---

### Step 1: 下载模型（使用国内源）⭐⭐⭐⭐⭐

**时间**: 2-4小时（取决于网络）

**操作步骤**:

1. **方法1: 使用脚本（推荐）**:
```bash
bash setup_models_china.sh
```

2. **方法2: 直接使用Python脚本**:
```bash
python download_models_china.py --model all --use-modelscope
```

**下载的模型**:
- `./models/Qwen2.5-3B-Instruct/` (约6GB)
- `./models/Llama3-8B-Instruct/` (约16GB)
- `./models/Gemma2-9B-Instruct/` (约18GB)

**验证模型完整性**:
```bash
# 检查每个模型
python check_model_integrity.py ./models/Qwen2.5-3B-Instruct
python check_model_integrity.py ./models/Llama3-8B-Instruct
python check_model_integrity.py ./models/Gemma2-9B-Instruct
```

**预期输出**: 每个模型显示 "Model integrity check passed"

**注意事项**:
- 确保有足够的磁盘空间（至少50GB）
- 如果下载中断，可以重新运行脚本继续下载
- 国内用户建议使用ModelScope镜像源

---

### Step 2: Baseline方法已实现 ✅

**状态**: Baseline方法已经实现完成

**实现位置**:
- `StreamingLLM_GPE/baselines/h2o_cache.py` - H2O baseline实现
- `StreamingLLM_GPE/baselines/streamingllm_cache.py` - StreamingLLM baseline实现

**测试Baseline**:

1. **测试H2O baseline**:
```bash
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_h2o \
    --h2o_budget 2048 \
    --output_dir ./output_logs/h2o_test \
    --max_samples 5 \
    --quantization 4bit
```

2. **测试StreamingLLM baseline**:
```bash
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_streamingllm \
    --streamingllm_window 512 \
    --output_dir ./output_logs/streamingllm_test \
    --max_samples 5 \
    --quantization 4bit
```

**预期输出**: 
- 生成 `./output_logs/h2o_test/results.json`
- 生成 `./output_logs/streamingllm_test/results.json`
- 日志文件显示推理过程无错误

**为什么最重要**: A级论文必须与SOTA方法对比，证明方法的优势

---

### Step 3: 运行A级论文实验（必须）⭐⭐⭐⭐⭐

**时间**: 2-3天（取决于硬件配置）

**操作步骤**:

1. **运行完整实验脚本**:
```bash
bash run_a_level_experiments.sh
```

**实验包含的4个阶段**:

#### Phase 1: 长序列内存效率对比

**测试序列长度**: 2000, 5000, 10000, 20000 tokens

**对比方法**:
1. Baseline (GPE) - 原始方法
2. H2O Baseline - 统一压缩方法
3. StreamingLLM Baseline - 滑动窗口方法
4. Head-Aware - Head-Aware方法（核心创新）

**每个方法运行**:
- 样本数: 100 samples
- 输出目录: `./output_logs/a_level_paper/long_seq_{长度}/{方法名}/`

**预期时间**: 每个序列长度 × 4种方法 × 100样本 ≈ 2-3小时

#### Phase 2: 预算影响分析

**测试预算**: 2048, 4096, 8192 tokens/layer

**运行方法**: Head-Aware

**输出目录**: `./output_logs/a_level_paper/budget_{预算}/`

**预期时间**: 3个预算 × 100样本 ≈ 1-2小时

#### Phase 3: 消融实验

**测试序列长度**: 5000 tokens

**对比配置**:
1. Baseline (GPE only) - 无压缩
2. Head-Aware - Head-Aware方法（核心创新）

**输出目录**: `./output_logs/a_level_paper/ablation/{配置名}/`

**预期时间**: 4种配置 × 100样本 ≈ 1-2小时

#### Phase 4: 结果分析和可视化

**自动运行**:
- 分析长序列实验结果
- 分析消融实验结果
- 生成可视化图表

**输出文件**:
- `./output_logs/a_level_paper/long_seq_10000_summary.csv`
- `./output_logs/a_level_paper/ablation_summary.csv`
- `./output_logs/a_level_paper/figures/` (可视化图表)

**预期时间**: 10-30分钟

**总预期时间**: 4-8小时（取决于硬件）

**注意事项**:
- 如果显存不足，可以减小 `MAX_SAMPLES` 或使用 `--quantization 4bit`
- 如果某个实验失败，脚本会继续运行其他实验
- 可以单独运行某个Phase，修改脚本中的注释

---

### Step 4: 多模型验证（必须）⭐⭐⭐⭐⭐

**时间**: 3-5天（取决于模型数量和硬件）

**操作步骤**:

1. **运行多模型实验脚本**:
```bash
bash run_multi_model_experiments.sh
```

**验证的模型**:
- Qwen2.5-3B-Instruct
- Llama3-8B-Instruct  
- Gemma2-9B-Instruct

**每个模型运行**:
- 长序列测试 (2000, 5000, 10000 tokens)
- Baseline对比 (H2O, StreamingLLM)
- Head-Aware方法

**输出目录**: `./output_logs/multi_model/{模型名}/`

**预期时间**: 每个模型约1-2天

**为什么必须**: 证明方法不依赖特定模型架构，具有通用性

---

### Step 5: 结果分析和论文准备

**时间**: 1-2天

**操作步骤**:

1. **分析实验结果**:
```bash
# 分析长序列实验结果
python analyze_experiment_results.py \
    --output_dir ./output_logs/a_level_paper/long_seq_10000 \
    --detailed \
    --save_csv ./output_logs/summary.csv \
    --save_json ./output_logs/summary.json \
    --save_latex ./output_logs/table.tex

# 分析消融实验结果
python analyze_experiment_results.py \
    --output_dir ./output_logs/a_level_paper/ablation \
    --detailed \
    --save_csv ./output_logs/ablation_summary.csv \
    --save_json ./output_logs/ablation_summary.json \
    --save_latex ./output_logs/ablation_table.tex
```

2. **生成可视化图表**:
```bash
python visualize_results.py \
    --results_dir ./output_logs/a_level_paper \
    --output_dir ./output_logs/figures \
    --include_budget
```

**输出文件**:
- CSV格式: 便于Excel分析
- JSON格式: 便于程序处理
- LaTeX格式: 直接用于论文表格
- 图表: PNG/PDF格式，用于论文插图

**预期输出**:
- 内存使用对比图表
- 性能（BLEU）对比图表
- 预算影响分析图表
- 消融实验结果表格

---

## 📊 A级论文成功标准

### 必须达到

1. ✅ **Baseline对比**: 与H2O和StreamingLLM对比
2. ✅ **内存效率**: 在10000+ tokens上减少40%+内存
3. ✅ **性能保持**: 性能损失 < 5%
4. ✅ **长度支持**: 支持20000+ tokens
5. ✅ **消融实验**: 证明各组件贡献
6. ✅ **多模型验证**: 在3个模型上验证

---

## 📁 项目文件结构

```
StreamingLLM/
├── StreamingLLM_GPE/              # 核心代码
│   ├── models/                    # 模型实现
│   │   ├── Qwen2_5/              # Qwen模型实现
│   │   ├── Llama3/               # Llama模型实现
│   │   └── Gemma2/               # Gemma模型实现
│   ├── evaluate/                  # 评估脚本
│   │   ├── multi_model_eval.py   # 多模型评估（主要脚本）⭐
│   │   ├── head_aware_eval.py    # Head-Aware评估
│   │   └── streaming_eval.py     # Streaming评估
│   ├── utils/                     # 工具函数
│   │   ├── head_analyzer.py      # Head分析器
│   │   ├── group_tracker.py     # Group跟踪器
│   │   └── budget_monitor.py    # 预算监控
│   ├── baselines/                 # Baseline实现 ⭐
│   │   ├── h2o_cache.py          # H2O baseline
│   │   └── streamingllm_cache.py # StreamingLLM baseline
│   └── configs/                   # 配置文件
│       └── params_qwen_inference.json
├── models/                        # 模型文件（需要下载）
│   ├── Qwen2.5-3B-Instruct/      # Qwen模型
│   ├── Llama3-8B-Instruct/       # Llama模型
│   └── Gemma2-9B-Instruct/       # Gemma模型
├── data_raw/                      # 原始数据
├── output_logs/                   # 实验结果输出目录
│   ├── a_level_paper/            # A级论文实验结果
│   └── multi_model/              # 多模型实验结果
├── requirements.txt               # Python依赖
├── setup_models_china.sh          # 模型下载脚本（国内源）⭐
├── download_models_china.py       # 模型下载Python脚本（国内源）⭐
├── run_a_level_experiments.sh     # A级论文实验脚本 ⭐
├── run_multi_model_experiments.sh # 多模型实验脚本
├── analyze_experiment_results.py  # 结果分析脚本
├── visualize_results.py          # 可视化脚本
├── check_environment.py           # 环境检查
├── check_model_integrity.py      # 模型检查
├── A_LEVEL_PAPER_FINAL_GUIDE.md   # 本文件（完整实验指南）⭐
├── BASELINE_IMPLEMENTATION_GUIDE.md # Baseline实现指南
└── README.md                      # 项目说明
```

---

## 🚀 立即开始（快速流程）

### 第一步：环境准备和下载模型

```bash
# 1. 检查环境
python check_environment.py

# 2. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 下载模型（使用国内源）
bash setup_models_china.sh

# 4. 验证模型
python check_model_integrity.py ./models/Qwen2.5-3B-Instruct
```

### 第二步：测试Baseline（已实现）

```bash
# 测试H2O baseline
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_h2o \
    --h2o_budget 2048 \
    --output_dir ./output_logs/h2o_test \
    --max_samples 5 \
    --quantization 4bit

# 测试StreamingLLM baseline
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_streamingllm \
    --streamingllm_window 512 \
    --output_dir ./output_logs/streamingllm_test \
    --max_samples 5 \
    --quantization 4bit
```

### 第三步：运行完整实验

```bash
# 运行A级论文完整实验（包含4个Phase）
bash run_a_level_experiments.sh
```

**实验时间**: 约4-8小时（取决于硬件配置）

### 第四步：分析结果

```bash
# 分析实验结果
python analyze_experiment_results.py \
    --output_dir ./output_logs/a_level_paper/long_seq_10000 \
    --detailed \
    --save_csv ./output_logs/summary.csv

# 生成可视化
python visualize_results.py \
    --results_dir ./output_logs/a_level_paper \
    --output_dir ./output_logs/figures
```

---

## ⚠️ 重要提示

1. **先下载模型**：所有实验都需要模型
2. **必须先实现Baseline**：没有baseline对比，无法证明方法优势
3. **消融实验是必须的**：A级论文必须证明各组件贡献
4. **多模型验证是必须的**：至少3个模型

---

## 📅 时间表

- **Day 1**: 下载模型（2-4小时）
- **Day 1-2**: 实现Baseline（1-2天）
- **Day 3-5**: 运行A级论文实验（2-3天）
- **Day 6-10**: 多模型验证（3-5天）
- **Day 11-12**: 结果分析（1-2天）
- **Week 3-6**: 论文撰写

---

## ✅ 检查清单

### 环境准备
- [ ] Python 3.8+ 已安装
- [ ] CUDA环境配置正确（如果使用GPU）
- [ ] 依赖包已安装 (`pip install -r requirements.txt`)
- [ ] ModelScope已安装（用于国内下载）

### 模型下载
- [ ] Qwen2.5-3B-Instruct 已下载并验证
- [ ] Llama3-8B-Instruct 已下载并验证（可选，用于多模型验证）
- [ ] Gemma2-9B-Instruct 已下载并验证（可选，用于多模型验证）

### Baseline实现
- [x] H2O baseline 已实现 (`StreamingLLM_GPE/baselines/h2o_cache.py`)
- [x] StreamingLLM baseline 已实现 (`StreamingLLM_GPE/baselines/streamingllm_cache.py`)
- [ ] H2O baseline 测试通过
- [ ] StreamingLLM baseline 测试通过

### A级论文实验
- [ ] Phase 1: 长序列内存效率对比实验完成
  - [ ] 2000 tokens测试
  - [ ] 5000 tokens测试
  - [ ] 10000 tokens测试
  - [ ] 20000 tokens测试
- [ ] Phase 2: 预算影响分析完成
- [ ] Phase 3: 消融实验完成
- [ ] Phase 4: 结果分析和可视化完成

### 多模型验证（可选但推荐）
- [ ] Qwen模型验证完成
- [ ] Llama模型验证完成
- [ ] Gemma模型验证完成

### 结果分析
- [ ] 实验结果已分析
- [ ] 可视化图表已生成
- [ ] 论文数据表格已准备（CSV/LaTeX格式）

### 论文准备
- [ ] 实验数据已整理
- [ ] 图表已优化
- [ ] 结果已与baseline对比
- [ ] 消融实验结果已分析

