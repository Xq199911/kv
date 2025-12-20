# Phase 0: Oracle Batch验证
# 目的：确定模型在非流式情况下的最高性能上限（Upper Bound）

param(
    [string]$ConfigFile = ".\scripts\windows\config.ps1"
)

$ErrorActionPreference = "Stop"

# 加载共享配置
if (Test-Path $ConfigFile) {
    . $ConfigFile
} else {
    Write-Host "ERROR: Config file not found: $ConfigFile" -ForegroundColor Red
    exit 1
}

Write-Host "========================================="
Write-Host "Phase 0: Oracle Batch Validation"
Write-Host "========================================="
Write-Host "Purpose: Determine upper bound performance"
Write-Host "Model: $MODEL_NAME"
Write-Host "Output: $BASE_OUTPUT_DIR\oracle_batch"
Write-Host ""

# 检查模型
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model not found at: $MODEL_PATH" -ForegroundColor Red
    Write-Host "Please download models first: .\scripts\windows\download_models.ps1" -ForegroundColor Yellow
    exit 1
}

# 创建输出目录
$output_dir = "$BASE_OUTPUT_DIR\oracle_batch"
New-Item -ItemType Directory -Force -Path $output_dir | Out-Null

# 运行Batch模式验证
Write-Host "Running Batch Mode Validation..."
$batch_test_len = 2000
$min_length_batch = [math]::Floor($batch_test_len / 100)

python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone $MODEL_NAME `
    --LLM_path $MODEL_PATH `
    --inference_mode batch `
    --output_dir $output_dir `
    --params $PARAMS `
    --min_source_length $min_length_batch `
    --max_samples 10 `
    --max_new_tokens $MAX_NEW_TOKENS

Write-Host ""
Write-Host "[OK] Phase 0 completed!" -ForegroundColor Green
Write-Host "Check BLEU score in: $output_dir\multi_model_eval.log"
Write-Host "If Batch BLEU is low (<20), the model or data has issues." -ForegroundColor Yellow
Write-Host ""

