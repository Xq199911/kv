# Phase 2: 预算影响分析
# 测试不同KV cache预算对Head-Aware方法的影响

param(
    [string]$ConfigFile = ".\scripts\windows\config.ps1",
    [int[]]$Budgets = @(),  # 如果为空，使用配置文件中的默认值
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
if ($Budgets.Count -eq 0) {
    $Budgets = $BUDGETS
}
if ($Samples -eq 0) {
    $Samples = $MAX_SAMPLES
}

Write-Host "========================================="
Write-Host "Phase 2: Budget Impact Analysis"
Write-Host "========================================="
Write-Host "Model: $MODEL_NAME"
Write-Host "Method: Head-Aware"
Write-Host "Budgets: $($Budgets -join ', ') tokens/layer"
Write-Host "Samples: $Samples"
Write-Host "Output: $BASE_OUTPUT_DIR\budget_*"
Write-Host ""

# 检查模型
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model not found at: $MODEL_PATH" -ForegroundColor Red
    exit 1
}

# 创建输出目录
New-Item -ItemType Directory -Force -Path $BASE_OUTPUT_DIR | Out-Null

$total_experiments = $Budgets.Count
$current_experiment = 0

foreach ($budget in $Budgets) {
    $current_experiment++
    Write-Host "----------------------------------------"
    Write-Host "[$current_experiment/$total_experiments] Testing budget: $budget tokens/layer"
    Write-Host "----------------------------------------"

    $output_dir = "$BASE_OUTPUT_DIR\budget_$budget"
    New-Item -ItemType Directory -Force -Path $output_dir | Out-Null

    python StreamingLLM_GPE/evaluate/multi_model_eval.py `
        --LLM_backbone $MODEL_NAME `
        --LLM_path $MODEL_PATH `
        --inference_mode streaming `
        --wait_k $WAIT_K `
        --use_head_aware `
        --analyze_heads `
        --total_budget $budget `
        --max_memory_gb $MAX_MEMORY_GB `
        --output_dir $output_dir `
        --params $PARAMS `
        --min_source_length $MIN_SOURCE_LENGTH `
        --max_samples $Samples `
        --max_new_tokens $MAX_NEW_TOKENS

    Write-Host ""
}

Write-Host ""
Write-Host "[OK] Phase 2 completed!" -ForegroundColor Green
Write-Host "Results saved to: $BASE_OUTPUT_DIR\budget_*"
Write-Host ""

