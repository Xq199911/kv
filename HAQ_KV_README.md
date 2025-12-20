# HAQ-KV: Head-Aware Quantized Key-Value Cache

**核心创新**: Head-Aware Heterogeneous Quantization（头感知异构量化）

## 🎯 核心创新

**"Not all memories require the same precision."** (不是所有的记忆都需要相同的精度)

HAQ-KV (Head-Aware Quantized Key-Value Cache) 或 Hi-KV (Hierarchical Knowledge Virtualization) 是一种基于异构量化的KV Cache优化方法。

## 💡 核心洞察

现有的SOTA方法（如H2O, StreamingLLM）本质上是在做"Selection & Eviction"（选择与驱逐），即认为某些Token不重要就直接丢弃。这对于需要精确匹配的"归纳头(Induction Heads)"或许有效，但对于负责长距离语义检索的"检索头(Retrieval Heads)"则是毁灭性的。

我们发现：
- **检索头的注意力分布往往比较稀疏且模糊**
- **它们更依赖于向量的方向（语义一致性）而非精确的模长**
- **因此，与其丢弃它们，不如模糊地记住它们**

## 🔬 方法论

### Head-Aware Heterogeneous Quantization (头感知异构量化)

根据Attention Head的功能特性，动态分配存储精度：

#### 1. Retrieval Heads (检索头) - "Low-Bit Semantic Retention"

- **量化策略**: INT4 或 INT2（激进量化）
- **存储内容**: 长期语义记忆
- **优势**: 
  - 在相同显存预算下，可以存储 **4倍到8倍** 的历史长度
  - 模型不再需要"遗忘"之前的关键实体
  - 只是记得稍微"模糊"了一点，但这足以支持 Needle-in-a-Haystack（大海捞针）式的检索

#### 2. Induction & Local Heads (归纳头与局部头) - "High-Precision Syntactic Buffer"

- **量化策略**: FP16/BF16（高精度）
- **存储内容**: 语法缓冲和上下文模式
- **原因**: 
  - 这部分Head负责维持语言的流利度和复制上下文模式（如N-gram）
  - 对数值精度高度敏感（微小的误差会导致生成的文本不通顺）
  - 采用改进的滑动窗口与重要性采样策略（Sink + Heavy Hitters）

## 🚀 使用方法

### 基本使用

```bash
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_haq_kv \
    --retrieval_bits 4 \
    --induction_bits 16 \
    --local_bits 16 \
    --total_budget 2048 \
    --analyze_heads \
    --output_dir ./output_logs/haq_kv_test \
    --max_samples 10
```

### 参数说明

- `--use_haq_kv`: 启用HAQ-KV方法
- `--retrieval_bits`: Retrieval Heads的量化位数 (2, 4, 8)
  - `2`: INT2，最激进，可存储8倍历史
  - `4`: INT4，平衡选择，可存储4倍历史（推荐）
  - `8`: INT8，保守选择，可存储2倍历史
- `--induction_bits`: Induction Heads的量化位数（通常保持16=FP16）
- `--local_bits`: Local Heads的量化位数（通常保持16=FP16）
- `--total_budget`: KV cache预算（基于FP16计算）
- `--analyze_heads`: 启用head功能分析（必须）

## 📊 预期效果

### 内存效率

- **相同显存下**: 可存储4-8倍的历史长度
- **Retrieval Heads**: 内存占用减少75%（INT4）或87.5%（INT2）
- **整体内存**: 根据head分布，平均节省40-60%

### 性能提升

- **无限上下文感知力**: 在显存占用不增加的情况下，有效回溯长度提升数倍
- **准确率保持**: 解决了H2O在长文本问答中因为误删Token导致准确率崩盘的问题
- **推理速度**: 保持极高的推理速度（量化操作开销很小）

## 🔍 技术细节

### 量化实现

量化器 (`StreamingLLM_GPE/utils/quantization.py`):
- 支持INT2, INT4, INT8量化
- 使用scale-based量化（每个head独立scale）
- 自动反量化用于attention计算

### Cache实现

HAQ-KV Cache (`StreamingLLM_GPE/models/Qwen2_5/haq_kv_cache.py`):
- 继承自 `DynamicCache`
- 根据head类型选择量化策略
- 存储量化后的KV，计算时反量化

### Head分析

使用现有的 `HeadAnalyzer` 进行head功能分类：
- Retrieval Heads: 高entropy，低局部性
- Induction Heads: 模式匹配特征
- Local Heads: 高局部性

## 📈 对比实验

### vs H2O

- **优势**: 不丢弃tokens，保留语义信息
- **劣势**: 量化可能引入微小误差（但对检索任务影响小）

### vs StreamingLLM

- **优势**: 不限制窗口大小，可以回溯更远的历史
- **劣势**: 需要head分析（一次性开销）

### vs Head-Aware (Eviction)

- **优势**: 存储更多历史，避免信息丢失
- **劣势**: 量化引入误差（但对检索头影响小）

## 🎓 论文贡献

1. **核心洞察**: "Not all memories require the same precision"
2. **方法论**: Head-Aware Heterogeneous Quantization
3. **实验验证**: 在多个模型和任务上验证有效性
4. **理论分析**: 量化对检索任务的影响分析

## 📝 引用

如果使用本方法，请引用：

```bibtex
@article{haqkv2025,
  title={HAQ-KV: Head-Aware Quantized Key-Value Cache for Efficient Long-Sequence Inference},
  author={...},
  journal={...},
  year={2025}
}
```

## 🔧 故障排除

### 问题1: 量化后BLEU下降

**原因**: 可能量化过于激进，或head分类不准确

**解决**:
- 增加 `retrieval_bits` 到 8
- 检查head分析结果
- 验证head分类是否正确

### 问题2: 内存节省不明显

**原因**: Retrieval Heads数量较少

**解决**:
- 检查head分析统计
- 调整head分类阈值
- 考虑使用更激进的量化（INT2）

### 问题3: 推理速度变慢

**原因**: 量化/反量化开销

**解决**:
- 使用INT4而不是INT2（减少量化次数）
- 批量处理量化操作
- 考虑使用硬件加速的量化

## 📚 相关文件

- `StreamingLLM_GPE/utils/quantization.py`: 量化工具
- `StreamingLLM_GPE/models/Qwen2_5/haq_kv_cache.py`: HAQ-KV实现
- `StreamingLLM_GPE/utils/head_analyzer.py`: Head分析器
- `StreamingLLM_GPE/evaluate/multi_model_eval.py`: 评估脚本

## 🎯 未来工作

1. 支持更多量化格式（FP8, BF8等）
2. 自适应量化位数选择
3. 硬件加速的量化操作
4. 更多模型的实现（Llama, Gemma等）

