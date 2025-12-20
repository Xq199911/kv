# Head-Aware 实验一键运行脚本

## 功能说明

这个脚本用于快速验证 Head-Aware 方法的两个实现：
1. **Head-Aware only** - 仅使用 Head-Aware 方法
2. **Head-Aware + Group-Aware (Full)** - 使用 Head-Aware + Group-Aware 完整方法

## 使用方法

### Python 版本（跨平台）

```bash
# 基本用法（快速验证，10个样本）
python scripts/run_head_aware_experiments.py --model-path ./models/Qwen2.5-3B-Instruct

# 完整验证（100个样本）
python scripts/run_head_aware_experiments.py --model-path ./models/Qwen2.5-3B-Instruct --max-samples 100

# 自定义预算
python scripts/run_head_aware_experiments.py --model-path ./models/Qwen2.5-3B-Instruct --total-budget 4096

# 自定义量化策略
python scripts/run_head_aware_experiments.py --model-path ./models/Qwen2.5-3B-Instruct --quantization 8bit
```

### Windows PowerShell 版本

```powershell
# 基本用法
# 如果需要修改配置，编辑脚本中的变量：
# $MAX_SAMPLES = 10        # 样本数
# $TOTAL_BUDGET = 2048     # 预算
# $QUANTIZATION = "4bit"   # 量化策略
```

## 参数说明

### Python 版本参数

- `--model-path`: 模型路径（必需）
- `--output-dir`: 输出目录（默认: `./output_logs/head_aware_experiments`）
- `--max-samples`: 最大样本数（默认: 10）
- `--total-budget`: KV cache预算 per layer（默认: 2048）
- `--quantization`: 量化策略，可选 `4bit`, `8bit`, `none`（默认: `4bit`）
- `--device`: GPU设备ID（默认: 0）
- `--min-source-length`: 最小源序列长度 words（默认: 3000）
- `--skip-check`: 跳过环境检查

## 输出结果

脚本会在输出目录下创建两个子目录：

```
output_logs/head_aware_experiments/
├── head_aware_only/
│   ├── results.json          # 详细实验结果
│   └── multi_model_eval.log  # 运行日志
├── head_aware_group_aware/
│   ├── results.json
│   └── multi_model_eval.log
└── comparison.json            # 对比结果（Python版本）
```

## 结果分析

### 关键指标

- **BLEU分数**: 翻译质量指标
- **峰值内存**: GPU峰值内存使用
- **Cache内存**: KV cache内存使用
- **平均延迟 (AL)**: 流式推理的平均延迟
- **延迟比例 (LAAL)**: 延迟比例

### 对比分析

脚本会自动对比两个实验的结果，显示：
- BLEU分数提升/下降
- 内存使用变化
- 延迟变化

## 示例输出

```
=========================================
实验结果对比
=========================================
指标                      Head-Aware          Head-Aware+Group-Aware    提升
-----------------------------------------------------------------------------------
BLEU分数                  0.4523              0.4689                    +3.67%
峰值内存 (GB)             3.45                3.38                      -2.03%
峰值Cache内存 (GB)         0.1234              0.1189                    -3.65%
平均延迟 (AL)             12.34               11.89                     -3.65%

总结:
  ✅ Full方法BLEU提升 3.67%
  ✅ Full方法内存减少 2.03%
```

## 下一步

如果实验结果满意：

1. **增加样本数进行更全面验证**:
   ```bash
   python scripts/run_head_aware_experiments.py --model-path ./models/Qwen2.5-3B-Instruct --max-samples 100
   ```

2. **运行完整实验**:
   ```bash
   bash run_a_level_experiments.sh
   # 或 Windows
   .\scripts\windows\run_a_level_experiments.ps1
   ```

## 注意事项

1. **首次运行建议使用小样本数**（如 10 个样本）快速验证
2. **确保模型已下载**并路径正确
3. **检查GPU显存**是否足够（建议至少 8GB）
4. **使用 4bit 量化**可以显著减少显存使用
5. **实验可能需要较长时间**，请耐心等待

## 故障排除

### 模型路径不存在
- 检查模型是否已下载
- 确认路径是否正确（相对路径或绝对路径）

### 显存不足
- 使用 `--quantization 4bit` 或 `8bit`
- 减少 `--max-samples`
- 减少 `--total-budget`

### 实验运行失败
- 查看日志文件：`output_logs/head_aware_experiments/*/multi_model_eval.log`
- 检查环境是否正确配置
- 确认所有依赖已安装

## 联系

如有问题，请查看主项目的 README 或 EXPERIMENT_GUIDE.md

