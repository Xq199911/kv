# Phase 1: 长序列内存效率对比实验
# 对比方法: Baseline (GPE), H2O, StreamingLLM, Head-Aware

param(
    [string]$ConfigFile = ".\scripts\windows\config.ps1",
    [int[]]$SequenceLengths = @(),  # 如果为空，使用配置文件中的默认值
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
if ($SequenceLengths.Count -eq 0) {
    $SequenceLengths = $LONG_SEQUENCE_LENGTHS
}
if ($Samples -eq 0) {
    $Samples = $MAX_SAMPLES
}

Write-Host "========================================="
Write-Host "Phase 1: Long Sequence Memory Efficiency"
Write-Host "========================================="
Write-Host "Model: $MODEL_NAME"
Write-Host "Sequence Lengths: $($SequenceLengths -join ', ')"
Write-Host "Samples per method: $Samples"
Write-Host "Output: $BASE_OUTPUT_DIR\long_seq_*"
Write-Host ""

# 检查模型
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model not found at: $MODEL_PATH" -ForegroundColor Red
    exit 1
}

# 创建输出目录
New-Item -ItemType Directory -Force -Path $BASE_OUTPUT_DIR | Out-Null

$total_experiments = $SequenceLengths.Count * 4  # 4种方法
$current_experiment = 0

foreach ($seq_len in $SequenceLengths) {
    Write-Host "----------------------------------------"
    Write-Host "Testing sequence length: $seq_len tokens"
    Write-Host "----------------------------------------"

    # 计算min_source_length
    $min_length = [math]::Floor($seq_len / 100)
    if ($min_length -lt 10) { $min_length = 10 }
    if ($min_length -gt 50) { $min_length = 50 }

    $seq_output_dir = "$BASE_OUTPUT_DIR\long_seq_$seq_len"
    New-Item -ItemType Directory -Force -Path $seq_output_dir | Out-Null

    # 1. Baseline (GPE)
    $current_experiment++
    Write-Host "[$current_experiment/$total_experiments] Baseline (GPE)..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py `
        --LLM_backbone $MODEL_NAME `
        --LLM_path $MODEL_PATH `
        --inference_mode streaming `
        --quantization none `
        --wait_k $WAIT_K `
        --output_dir "$seq_output_dir\baseline" `
        --params $PARAMS `
        --min_source_length $min_length `
        --max_samples $Samples `
        --max_new_tokens $MAX_NEW_TOKENS

    # 2. H2O Baseline
    $current_experiment++
    if (Test-Path "StreamingLLM_GPE\baselines\h2o_cache.py") {
        Write-Host "[$current_experiment/$total_experiments] H2O Baseline..."
        python StreamingLLM_GPE/evaluate/multi_model_eval.py `
            --LLM_backbone $MODEL_NAME `
            --LLM_path $MODEL_PATH `
            --inference_mode streaming `
            --quantization none `
            --wait_k $WAIT_K `
            --use_h2o `
            --h2o_budget $TOTAL_BUDGET `
            --max_memory_gb $MAX_MEMORY_GB `
            --output_dir "$seq_output_dir\h2o" `
            --params $PARAMS `
            --min_source_length $min_length `
            --max_samples $Samples `
            --max_new_tokens $MAX_NEW_TOKENS
    } else {
        Write-Host "[$current_experiment/$total_experiments] Skipping H2O (not implemented)" -ForegroundColor Yellow
    }

    # 3. StreamingLLM Baseline
    $current_experiment++
    if (Test-Path "StreamingLLM_GPE\baselines\streamingllm_cache.py") {
        Write-Host "[$current_experiment/$total_experiments] StreamingLLM Baseline..."
        python StreamingLLM_GPE/evaluate/multi_model_eval.py `
            --LLM_backbone $MODEL_NAME `
            --LLM_path $MODEL_PATH `
            --inference_mode streaming `
            --quantization none `
            --wait_k $WAIT_K `
            --use_streamingllm `
            --streamingllm_window $STREAMINGLLM_WINDOW `
            --max_memory_gb $MAX_MEMORY_GB `
            --output_dir "$seq_output_dir\streamingllm" `
            --params $PARAMS `
            --min_source_length $min_length `
            --max_samples $Samples `
            --max_new_tokens $MAX_NEW_TOKENS
    } else {
        Write-Host "[$current_experiment/$total_experiments] Skipping StreamingLLM (not implemented)" -ForegroundColor Yellow
    }

    # 4. Head-Aware
    $current_experiment++
    Write-Host "[$current_experiment/$total_experiments] Head-Aware..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py `
        --LLM_backbone $MODEL_NAME `
        --LLM_path $MODEL_PATH `
        --inference_mode streaming `
        --quantization none `
        --wait_k $WAIT_K `
        --use_head_aware `
        --analyze_heads `
        --total_budget $TOTAL_BUDGET `
        --max_memory_gb $MAX_MEMORY_GB `
        --output_dir "$seq_output_dir\head_aware" `
        --params $PARAMS `
        --min_source_length $min_length `
        --max_samples $Samples `
        --max_new_tokens $MAX_NEW_TOKENS

    Write-Host ""
}

Write-Host ""
Write-Host "[OK] Phase 1 completed!" -ForegroundColor Green
Write-Host "Results saved to: $BASE_OUTPUT_DIR\long_seq_*"
Write-Host ""

