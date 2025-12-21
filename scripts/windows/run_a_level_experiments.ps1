# 论文完整实验脚本 - Windows版本（已重构）
# 
# ⚠️ 注意：此脚本已重构为模块化脚本
# 推荐使用：.\scripts\windows\run_all_experiments.ps1
# 
# 如需单独运行某个Phase，使用：
# - phase0_oracle_batch.ps1
# - phase1_long_sequence.ps1
# - phase2_budget_analysis.ps1
# - phase3_ablation.ps1
# - phase4_analysis.ps1

$ErrorActionPreference = "Stop"

# ================= 配置区域 =================
$MODEL_NAME = "Qwen"
$MODEL_PATH = ".\models\Qwen2.5-3B-Instruct"
$BASE_OUTPUT_DIR = ".\output_logs\a_level_paper_fixed"
$PARAMS = ".\StreamingLLM_GPE\configs\params_qwen_inference.json"

#  放宽 Wait-k 约束，通用模型需要更多上下文
$WAIT_K = 15
#  增加生成长度上限，防止长序列被截断
$MAX_NEW_TOKENS = 4096

$TOTAL_BUDGET = 2048
$MAX_MEMORY_GB = 20.0

# 实验配置
$MAX_SAMPLES = 100  # [FIXED] 与文档一致，从50改为100
$MIN_SOURCE_LENGTH = 20

# 长序列测试配置
$LONG_SEQUENCE_LENGTHS = @(2000, 5000, 10000, 20000)
$BUDGETS = @(2048, 4096, 8192)  # [FIXED] 与文档一致，从(512,1024,2048)改为(2048,4096,8192)

# Baseline参数配置（统一标准）
$STREAMINGLLM_WINDOW = 512  # [FIXED] 统一窗口大小，从256改为512
$SINK_TOKENS = 128  # [FIXED] 统一sink tokens数量

Write-Host "========================================="
Write-Host "A-Level Paper Experiment Suite (Windows - FIXED)"
Write-Host "========================================="
Write-Host "GPU: RTX 4090 (24GB)"
Write-Host "Model: $MODEL_NAME"
Write-Host "Wait-k: $WAIT_K (Increased from 5)"
Write-Host "Max New Tokens: $MAX_NEW_TOKENS (Increased from 2048)"
Write-Host "Output: $BASE_OUTPUT_DIR"
Write-Host ""

# 创建输出目录
New-Item -ItemType Directory -Force -Path $BASE_OUTPUT_DIR | Out-Null

# 检查模型是否存在
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model not found at: $MODEL_PATH"
    Write-Host "Please download models first: .\scripts\windows\download_models.ps1"
    exit 1
}

# ============================================
# Phase 0: Oracle Batch 验证
# 目的：确定模型在非流式情况下的最高性能上限
# ============================================
Write-Host ""
Write-Host "========================================="
Write-Host "Phase 0: Oracle Batch Validation (Upper Bound)"
Write-Host "========================================="
Write-Host ""

Write-Host "Running Batch Mode Validation..."
# 使用一个中等长度进行快速验证
$batch_test_len = 2000
$min_length_batch = [math]::Floor($batch_test_len / 100)

python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone $MODEL_NAME `
    --LLM_path $MODEL_PATH `
    --inference_mode batch `
    --output_dir "$BASE_OUTPUT_DIR\oracle_batch" `
    --params $PARAMS `
    --min_source_length $min_length_batch `
    --max_samples 10 `
    --max_new_tokens $MAX_NEW_TOKENS

Write-Host "Batch validation complete. Check BLEU score in $BASE_OUTPUT_DIR\oracle_batch\multi_model_eval.log"
Write-Host "If Batch BLEU is low (<20), the model or data has issues regardless of streaming settings."
Write-Host ""

# ============================================
# Phase 1: 长序列内存效率对比实验
# ============================================
Write-Host ""
Write-Host "========================================="
Write-Host "Phase 1: Long Sequence Memory Efficiency"
Write-Host "========================================="
Write-Host ""

foreach ($seq_len in $LONG_SEQUENCE_LENGTHS) {
    Write-Host "----------------------------------------"
    Write-Host "Testing sequence length: $seq_len tokens"
    Write-Host "----------------------------------------"

    # Calculate min_source_length
    $min_length = [math]::Floor($seq_len / 100)
    if ($min_length -lt 10) { $min_length = 10 }
    if ($min_length -gt 50) { $min_length = 50 }

    # 1. Baseline (GPE)
    Write-Host "[1/5] Running Baseline (GPE)..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py `
        --LLM_backbone $MODEL_NAME `
        --LLM_path $MODEL_PATH `
        --inference_mode streaming `
        --wait_k $WAIT_K `
        --output_dir "$BASE_OUTPUT_DIR\long_seq_$seq_len\baseline" `
        --params $PARAMS `
        --min_source_length $min_length `
        --max_samples $MAX_SAMPLES `
        --max_new_tokens $MAX_NEW_TOKENS

    # 2. H2O Baseline (if implemented)
    if (Test-Path "StreamingLLM_GPE\baselines\h2o_cache.py") {
        Write-Host "[2/5] Running H2O Baseline..."
        python StreamingLLM_GPE/evaluate/multi_model_eval.py `
            --LLM_backbone $MODEL_NAME `
            --LLM_path $MODEL_PATH `
            --inference_mode streaming `
            --wait_k $WAIT_K `
            --use_h2o `
            --h2o_budget $TOTAL_BUDGET `
            --max_memory_gb $MAX_MEMORY_GB `
            --output_dir "$BASE_OUTPUT_DIR\long_seq_$seq_len\h2o" `
            --params $PARAMS `
            --min_source_length $min_length `
            --max_samples $MAX_SAMPLES `
            --max_new_tokens $MAX_NEW_TOKENS
    } else {
        Write-Host "[2/5] Skipping H2O (not implemented yet)"
    }

    # 3. StreamingLLM Baseline (if implemented)
    if (Test-Path "StreamingLLM_GPE\baselines\streamingllm_cache.py") {
        Write-Host "[3/5] Running StreamingLLM Baseline..."
        python StreamingLLM_GPE/evaluate/multi_model_eval.py `
            --LLM_backbone $MODEL_NAME `
            --LLM_path $MODEL_PATH `
            --inference_mode streaming `
            --wait_k $WAIT_K `
            --use_streamingllm `
            --streamingllm_window $STREAMINGLLM_WINDOW `
            --max_memory_gb $MAX_MEMORY_GB `
            --output_dir "$BASE_OUTPUT_DIR\long_seq_$seq_len\streamingllm" `
            --params $PARAMS `
            --min_source_length $min_length `
            --max_samples $MAX_SAMPLES `
            --max_new_tokens $MAX_NEW_TOKENS
    } else {
        Write-Host "[3/5] Skipping StreamingLLM (not implemented yet)"
    }

    # 4. Head-Aware
    Write-Host "[4/4] Running Head-Aware..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py `
        --LLM_backbone $MODEL_NAME `
        --LLM_path $MODEL_PATH `
        --inference_mode streaming `
        --wait_k $WAIT_K `
        --use_head_aware `
        --analyze_heads `
        --total_budget $TOTAL_BUDGET `
        --max_memory_gb $MAX_MEMORY_GB `
        --output_dir "$BASE_OUTPUT_DIR\long_seq_$seq_len\head_aware" `
        --params $PARAMS `
        --min_source_length $min_length `
        --max_samples $MAX_SAMPLES `
        --max_new_tokens $MAX_NEW_TOKENS

    Write-Host ""
}

# ============================================
# Phase 2: 不同预算的影响
# ============================================
Write-Host ""
Write-Host "========================================="
Write-Host "Phase 2: Budget Impact Analysis"
Write-Host "========================================="
Write-Host ""

foreach ($budget in $BUDGETS) {
    Write-Host "----------------------------------------"
    Write-Host "Testing budget: $budget tokens/layer"
    Write-Host "----------------------------------------"

    python StreamingLLM_GPE/evaluate/multi_model_eval.py `
        --LLM_backbone $MODEL_NAME `
        --LLM_path $MODEL_PATH `
        --inference_mode streaming `
        --wait_k $WAIT_K `
        --use_head_aware `
        --analyze_heads `
        --total_budget $budget `
        --max_memory_gb $MAX_MEMORY_GB `
        --output_dir "$BASE_OUTPUT_DIR\budget_$budget" `
        --params $PARAMS `
        --min_source_length $MIN_SOURCE_LENGTH `
        --max_samples $MAX_SAMPLES `
        --max_new_tokens $MAX_NEW_TOKENS

    Write-Host ""
}

# ============================================
# Phase 3: 消融实验
# ============================================
Write-Host ""
Write-Host "========================================="
Write-Host "Phase 3: Ablation Study"
Write-Host "========================================="
Write-Host ""

$ABLATION_SEQ_LEN = 5000
Write-Host "Running ablation experiments with sequence length: $ABLATION_SEQ_LEN"
Write-Host ""

$min_length = [math]::Floor($ABLATION_SEQ_LEN / 100)
if ($min_length -lt 10) { $min_length = 10 }
if ($min_length -gt 50) { $min_length = 50 }

# 1. Baseline (GPE only)
Write-Host "1. Baseline (GPE only)..."
python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone $MODEL_NAME `
    --LLM_path $MODEL_PATH `
    --inference_mode streaming `
    --wait_k $WAIT_K `
    --output_dir "$BASE_OUTPUT_DIR\ablation\baseline" `
    --params $PARAMS `
    --min_source_length $min_length `
    --max_samples $MAX_SAMPLES `
    --max_new_tokens $MAX_NEW_TOKENS

# 2. Head-Aware
Write-Host "2. Head-Aware..."
python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone $MODEL_NAME `
    --LLM_path $MODEL_PATH `
    --inference_mode streaming `
    --wait_k $WAIT_K `
    --use_head_aware `
    --analyze_heads `
    --total_budget $TOTAL_BUDGET `
    --max_memory_gb $MAX_MEMORY_GB `
    --output_dir "$BASE_OUTPUT_DIR\ablation\head_aware" `
    --params $PARAMS `
    --min_source_length $min_length `
    --max_samples $MAX_SAMPLES `
    --max_new_tokens $MAX_NEW_TOKENS

# ============================================
# Phase 4: 结果分析和汇总
# ============================================
Write-Host ""
Write-Host "========================================="
Write-Host "Phase 4: Results Analysis"
Write-Host "========================================="
Write-Host ""

# 分析长序列实验结果
Write-Host "Analyzing long sequence results..."
python analyze_experiment_results.py `
    --output_dir "$BASE_OUTPUT_DIR\long_seq_10000" `
    --detailed `
    --save_csv "$BASE_OUTPUT_DIR\long_seq_10000_summary.csv" `
    --save_json "$BASE_OUTPUT_DIR\long_seq_10000_summary.json" `
    --save_latex "$BASE_OUTPUT_DIR\long_seq_10000_table.tex"

# 分析消融实验结果
Write-Host "Analyzing ablation results..."
python analyze_experiment_results.py `
    --output_dir "$BASE_OUTPUT_DIR\ablation" `
    --detailed `
    --save_csv "$BASE_OUTPUT_DIR\ablation_summary.csv" `
    --save_json "$BASE_OUTPUT_DIR\ablation_summary.json" `
    --save_latex "$BASE_OUTPUT_DIR\ablation_table.tex"

# 生成可视化
Write-Host "Generating visualizations..."
python visualize_results.py `
    --results_dir "$BASE_OUTPUT_DIR\long_seq_10000" `
    --output_dir "$BASE_OUTPUT_DIR\figures" `
    --include_budget

Write-Host ""
Write-Host "========================================="
Write-Host "All A-Level Experiments Completed!"
Write-Host "========================================="
Write-Host ""
Write-Host "Results saved to: $BASE_OUTPUT_DIR"