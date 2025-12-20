# Phase 5: HAQ-KV实验（主要目标）
# 测试Head-Aware Quantized KV Cache方法
# 这是论文的主要贡献：基于Head功能特性的异构量化KV Cache
#
# 使用方法：
#   .\scripts\windows\phase5_haq_kv.ps1
#   .\scripts\windows\phase5_haq_kv.ps1 -RetrievalBits 2  # 使用INT2量化

param(
    [string]$ConfigFile = ".\scripts\windows\config.ps1",
    [int]$RetrievalBits = 4,  # Retrieval Heads量化位数 (2, 4, 8)
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
if ($Samples -eq 0) {
    $Samples = $MAX_SAMPLES
}

Write-Host "========================================="
Write-Host "Phase 5: HAQ-KV Experiment"
Write-Host "========================================="
Write-Host "Method: HAQ-KV (Head-Aware Quantized KV Cache)"
Write-Host "Model: $MODEL_NAME"
Write-Host "Retrieval Bits: $RetrievalBits"
Write-Host "Samples: $Samples"
Write-Host "Output: $BASE_OUTPUT_DIR\haq_kv_*"
Write-Host ""
Write-Host "Note: This is the MAIN CONTRIBUTION of the paper."
Write-Host "      HAQ-KV uses heterogeneous quantization based on head functions."
Write-Host "      Retrieval Heads use INT$RetrievalBits, others use FP16."
Write-Host ""

# 检查模型
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model not found at: $MODEL_PATH" -ForegroundColor Red
    exit 1
}

# 创建输出目录
New-Item -ItemType Directory -Force -Path $BASE_OUTPUT_DIR | Out-Null

# 测试不同序列长度
$test_lengths = @(2000, 5000, 10000)

foreach ($seq_len in $test_lengths) {
    Write-Host "----------------------------------------"
    Write-Host "Testing sequence length: $seq_len tokens"
    Write-Host "----------------------------------------"
    
    # 计算min_source_length
    $min_length = [math]::Floor($seq_len / 100)
    if ($min_length -lt 10) { $min_length = 10 }
    if ($min_length -gt 50) { $min_length = 50 }
    
    $output_dir = "$BASE_OUTPUT_DIR\haq_kv_${RetrievalBits}bit\long_seq_$seq_len"
    New-Item -ItemType Directory -Force -Path $output_dir | Out-Null
    
    Write-Host "Running HAQ-KV with INT$RetrievalBits quantization..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py `
        --LLM_backbone $MODEL_NAME `
        --LLM_path $MODEL_PATH `
        --inference_mode streaming `
        --wait_k $WAIT_K `
        --use_haq_kv `
        --retrieval_bits $RetrievalBits `
        --induction_bits 16 `
        --local_bits 16 `
        --analyze_heads `
        --total_budget $TOTAL_BUDGET `
        --max_memory_gb $MAX_MEMORY_GB `
        --output_dir $output_dir `
        --params $PARAMS `
        --min_source_length $min_length `
        --max_samples $Samples `
        --max_new_tokens $MAX_NEW_TOKENS
    
    Write-Host ""
}

Write-Host ""
Write-Host "[OK] Phase 5 (HAQ-KV) completed!" -ForegroundColor Green
Write-Host "Results saved to: $BASE_OUTPUT_DIR\haq_kv_${RetrievalBits}bit\*"
Write-Host ""
Write-Host "Comparison:"
Write-Host "  - Compare with Phase 1 Head-Aware results"
Write-Host "  - Check memory savings vs performance trade-off"
Write-Host ""

