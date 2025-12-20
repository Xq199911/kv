# Phase 3: 消融实验
# 对比: Baseline (GPE) vs Head-Aware

param(
    [string]$ConfigFile = ".\scripts\windows\config.ps1",
    [int]$SequenceLength = 0,  # 如果为0，使用默认值5000
    [int]$Samples = 0  # 如果为0，使用配置文件中的默认值
)

$ErrorActionPreference = "Stop"

# 加载共享配置
if (Test-Path $ConfigFile) {
    . $ConfigFile
} else {
    Write-Host "ERROR: Config file not found: $ConfigFile" -ForegroundColor Red
    exit 1
}

# 使用参数或默认值
if ($SequenceLength -eq 0) {
    $SequenceLength = 5000
}
if ($Samples -eq 0) {
    $Samples = $MAX_SAMPLES
}

Write-Host "========================================="
Write-Host "Phase 3: Ablation Study"
Write-Host "========================================="
Write-Host "Model: $MODEL_NAME"
Write-Host "Sequence Length: $SequenceLength tokens"
Write-Host "Samples: $Samples"
Write-Host "Output: $BASE_OUTPUT_DIR\ablation\*"
Write-Host ""

# 检查模型
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model not found at: $MODEL_PATH" -ForegroundColor Red
    exit 1
}

# 创建输出目录
$ablation_dir = "$BASE_OUTPUT_DIR\ablation"
New-Item -ItemType Directory -Force -Path $ablation_dir | Out-Null

# 计算min_source_length
$min_length = [math]::Floor($SequenceLength / 100)
if ($min_length -lt 10) { $min_length = 10 }
if ($min_length -gt 50) { $min_length = 50 }

# 1. Baseline (GPE only)
Write-Host "[1/2] Baseline (GPE only)..."
python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone $MODEL_NAME `
    --LLM_path $MODEL_PATH `
    --inference_mode streaming `
    --wait_k $WAIT_K `
    --output_dir "$ablation_dir\baseline" `
    --params $PARAMS `
    --min_source_length $min_length `
    --max_samples $Samples `
    --max_new_tokens $MAX_NEW_TOKENS

Write-Host ""

# 2. Head-Aware
Write-Host "[2/2] Head-Aware..."
python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone $MODEL_NAME `
    --LLM_path $MODEL_PATH `
    --inference_mode streaming `
    --wait_k $WAIT_K `
    --use_head_aware `
    --analyze_heads `
    --total_budget $TOTAL_BUDGET `
    --max_memory_gb $MAX_MEMORY_GB `
    --output_dir "$ablation_dir\head_aware" `
    --params $PARAMS `
    --min_source_length $min_length `
    --max_samples $Samples `
    --max_new_tokens $MAX_NEW_TOKENS

Write-Host ""
Write-Host "[OK] Phase 3 completed!" -ForegroundColor Green
Write-Host "Results saved to: $ablation_dir\*"
Write-Host ""

