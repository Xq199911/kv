# Head-Aware Batch模式对比实验 - Windows版本
# 用于验证BLEU分数是否因推理模式不同而偏低

$ErrorActionPreference = "Stop"

# ================= 配置区域 =================
$MODEL_NAME = "Qwen"
$MODEL_PATH = ".\models\Qwen2.5-3B-Instruct"
$BASE_OUTPUT_DIR = ".\output_logs\head_aware_batch_comparison"
$PARAMS = ".\StreamingLLM_GPE\configs\params_qwen_inference.json"

$TOTAL_BUDGET = 2048
$MAX_MEMORY_GB = 20.0

# 实验配置
$MAX_SAMPLES = 10
$MIN_SOURCE_LENGTH = 50
$MAX_NEW_TOKENS = 4096

Write-Host "========================================="
Write-Host "Head-Aware Batch Mode Comparison"
Write-Host "========================================="
Write-Host "Model: $MODEL_NAME"
Write-Host "Model Path: $MODEL_PATH"
Write-Host "Output Dir: $BASE_OUTPUT_DIR"
Write-Host "Inference Mode: BATCH (for fair comparison)"
Write-Host "Min Source Length: $MIN_SOURCE_LENGTH words"
Write-Host "Max Samples: $MAX_SAMPLES"
Write-Host "Total Budget: $TOTAL_BUDGET tokens/layer"
Write-Host "Max New Tokens: $MAX_NEW_TOKENS"
Write-Host ""
Write-Host "NOTE: This uses BATCH mode to compare fairly with baseline."
Write-Host "      Baseline (H2O/StreamingLLM) also used batch mode."
Write-Host ""

# 检查模型是否存在
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model not found at: $MODEL_PATH" -ForegroundColor Red
    Write-Host "Please download models first: .\scripts\windows\download_models.ps1" -ForegroundColor Yellow
    exit 1
}

# 创建输出目录
New-Item -ItemType Directory -Force -Path $BASE_OUTPUT_DIR | Out-Null

# ================= Head-Aware Batch模式实验 =================
Write-Host ""
Write-Host "========================================="
Write-Host "Head-Aware Experiment (BATCH Mode)"
Write-Host "========================================="
Write-Host ""

$OUTPUT_DIR = Join-Path $BASE_OUTPUT_DIR "head_aware_batch"
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

Write-Host "Running Head-Aware in BATCH mode..."
python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone $MODEL_NAME `
    --LLM_path $MODEL_PATH `
    --inference_mode batch `
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
            $bleu = $results.bleu_score
            Write-Host "BLEU Score: $($bleu.ToString('F4'))" -ForegroundColor Cyan
            
            # 对比分析
            Write-Host ""
            Write-Host "Comparison:" -ForegroundColor Yellow
            Write-Host "  - Baseline (H2O/StreamingLLM, batch): ~17.89"
            Write-Host "  - Your Head-Aware (batch): $($bleu.ToString('F4'))"
            
            if ($bleu -lt 15) {
                Write-Host ""
                Write-Host "  ⚠️  BLEU still low. Possible reasons:" -ForegroundColor Yellow
                Write-Host "     1. KV cache budget too small"
                Write-Host "     2. Head-Aware strategy needs tuning"
                Write-Host "     3. Model needs fine-tuning for translation"
            } elseif ($bleu -ge 15) {
                Write-Host ""
                Write-Host "  ✓ BLEU score is reasonable for batch mode!" -ForegroundColor Green
            }
        }
        
        if ($results.memory_stats) {
            $mem = $results.memory_stats.peak_memory_gb
            if ($mem -gt 0) {
                Write-Host ""
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
Write-Host "1. Compare this BATCH result with your STREAMING result (7.44)"
Write-Host "2. If BATCH mode BLEU is much higher, the issue is streaming mode"
Write-Host "3. If BATCH mode BLEU is still low, check KV cache budget and strategy"
Write-Host "4. See BLEU_SCORE_ANALYSIS.md for detailed analysis"
Write-Host ""

