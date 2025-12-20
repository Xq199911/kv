# Head-Aware 实验一键运行脚本 - Windows版本
# 直接复制自完整实验脚本中的 Head-Aware 实验部分

$ErrorActionPreference = "Stop"

# ================= 配置区域 =================
$MODEL_NAME = "Qwen"
$MODEL_PATH = ".\models\Qwen2.5-3B-Instruct"
$BASE_OUTPUT_DIR = ".\output_logs\head_aware_experiments"
$PARAMS = ".\StreamingLLM_GPE\configs\params_qwen_inference.json"

# 放宽 Wait-k 约束，通用模型需要更多上下文
$WAIT_K = 15
# 增加生成长度上限，防止长序列被截断（恢复为原始设置）
$MAX_NEW_TOKENS = 4096

$TOTAL_BUDGET = 2048
$MAX_MEMORY_GB = 20.0

# 实验配置
$MAX_SAMPLES = 10
# 使用中等序列长度进行验证
$SEQ_LEN = 5000
$MIN_SOURCE_LENGTH = [math]::Floor($SEQ_LEN / 100)
if ($MIN_SOURCE_LENGTH -lt 10) { $MIN_SOURCE_LENGTH = 10 }
if ($MIN_SOURCE_LENGTH -gt 50) { $MIN_SOURCE_LENGTH = 50 }

Write-Host "========================================="
Write-Host "Head-Aware Experiment Runner"
Write-Host "========================================="
Write-Host "Model: $MODEL_NAME"
Write-Host "Model Path: $MODEL_PATH"
Write-Host "Output Dir: $BASE_OUTPUT_DIR"
Write-Host "Sequence Length: $SEQ_LEN tokens"
Write-Host "Min Source Length: $MIN_SOURCE_LENGTH words"
Write-Host "Max Samples: $MAX_SAMPLES"
Write-Host "Total Budget: $TOTAL_BUDGET tokens/layer"
Write-Host "Wait-k: $WAIT_K"
Write-Host "Max New Tokens: $MAX_NEW_TOKENS"
Write-Host ""

# 检查模型是否存在
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model not found at: $MODEL_PATH" -ForegroundColor Red
    Write-Host "Please download models first: .\scripts\windows\download_models.ps1" -ForegroundColor Yellow
    exit 1
}

# 创建输出目录
New-Item -ItemType Directory -Force -Path $BASE_OUTPUT_DIR | Out-Null

# ================= Head-Aware 实验 =================
Write-Host ""
Write-Host "========================================="
Write-Host "Head-Aware Experiment"
Write-Host "========================================="
Write-Host ""

$OUTPUT_DIR = Join-Path $BASE_OUTPUT_DIR "head_aware_only"
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

Write-Host "Running Head-Aware..."
python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone $MODEL_NAME `
    --LLM_path $MODEL_PATH `
    --inference_mode streaming `
    --wait_k $WAIT_K `
    --use_head_aware `
    --analyze_heads `
    --total_budget $TOTAL_BUDGET `
    --max_memory_gb $MAX_MEMORY_GB `
    --output_dir $OUTPUT_DIR `
    --params $PARAMS `
    --min_source_length $MIN_SOURCE_LENGTH `
    --max_samples $MAX_SAMPLES `
    --max_new_tokens $MAX_NEW_TOKENS

Write-Host ""
Write-Host "[OK] Experiment completed!" -ForegroundColor Green
Write-Host ""

# ================= 结果总结 =================
Write-Host ""
Write-Host "========================================="
Write-Host "Experiment Completed"
Write-Host "========================================="
Write-Host ""

Write-Host "Results Directory: $OUTPUT_DIR"
Write-Host ""

$results_file = Join-Path $OUTPUT_DIR "results.json"

if (Test-Path $results_file) {
    Write-Host "  [OK] Results file: $results_file" -ForegroundColor Green
    Write-Host ""
    
    # 尝试读取并显示结果摘要
    try {
        $results = Get-Content $results_file -Encoding UTF8 | ConvertFrom-Json
        
        Write-Host "========================================="
        Write-Host "Results Summary"
        Write-Host "========================================="
        Write-Host ""
        
        if ($results.bleu_score) {
            Write-Host "BLEU Score: $($results.bleu_score.ToString('F4'))"
        }
        
        if ($results.memory_stats) {
            $mem = $results.memory_stats.peak_memory_gb
            if ($mem -gt 0) {
                Write-Host "Peak Memory: $($mem.ToString('F2')) GB"
            }
        }
        
        Write-Host ""
    } catch {
        Write-Host "[INFO] Results file exists but cannot parse for summary" -ForegroundColor Yellow
        Write-Host "Please check the file manually: $results_file"
        Write-Host ""
    }
} else {
    Write-Host "  [WARNING] Results file not found: $results_file" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Next Steps:"
Write-Host "1. Check result file for detailed metrics: $results_file"
Write-Host "2. If results are good, run full experiments:"
Write-Host "   .\scripts\windows\run_a_level_experiments.ps1"
Write-Host "3. Or increase sample count for more comprehensive validation:"
Write-Host "   Edit `$MAX_SAMPLES variable in this script"
Write-Host ""
