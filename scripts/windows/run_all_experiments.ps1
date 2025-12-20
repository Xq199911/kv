# ============================================
# A级论文完整实验 - 一键运行脚本
# ============================================
# 
# 功能：自动运行所有Phase的实验（Phase 0-4）
# 
# 使用方法：
#   .\scripts\windows\run_all_experiments.ps1                    # 运行所有实验
#   .\scripts\windows\run_all_experiments.ps1 -SkipPhase0        # 跳过Phase 0
#   .\scripts\windows\run_all_experiments.ps1 -Phase1Only         # 只运行Phase 1
#   .\scripts\windows\run_all_experiments.ps1 -SkipPhase4        # 跳过结果分析
#
# ============================================

param(
    [string]$ConfigFile = ".\scripts\windows\config.ps1",
    [switch]$SkipPhase0 = $false,  # 跳过Phase 0（Oracle Batch验证）
    [switch]$SkipPhase4 = $false,  # 跳过Phase 4（结果分析）
    [switch]$Phase0Only = $false,  # 只运行Phase 0
    [switch]$Phase1Only = $false,  # 只运行Phase 1
    [switch]$Phase2Only = $false,  # 只运行Phase 2
    [switch]$Phase3Only = $false,  # 只运行Phase 3
    [switch]$Phase4Only = $false   # 只运行Phase 4
)

$ErrorActionPreference = "Stop"

# 加载共享配置
if (Test-Path $ConfigFile) {
    . $ConfigFile
} else {
    Write-Host "ERROR: Config file not found: $ConfigFile" -ForegroundColor Red
    Write-Host "Please check the path or create config.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "========================================="
Write-Host "A-Level Paper: Complete Experiment Suite"
Write-Host "========================================="
Write-Host "Model: $MODEL_NAME"
Write-Host "Model Path: $MODEL_PATH"
Write-Host "Output Directory: $BASE_OUTPUT_DIR"
Write-Host "Max Samples: $MAX_SAMPLES"
Write-Host "Total Budget: $TOTAL_BUDGET tokens/layer"
Write-Host ""

# 检查模型
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model not found at: $MODEL_PATH" -ForegroundColor Red
    Write-Host "Please download models first:" -ForegroundColor Yellow
    Write-Host "  .\scripts\windows\download_models.ps1" -ForegroundColor Yellow
    exit 1
}

# 创建输出目录
New-Item -ItemType Directory -Force -Path $BASE_OUTPUT_DIR | Out-Null

# 记录开始时间
$start_time = Get-Date
Write-Host "Start Time: $start_time"
Write-Host ""

# ============================================
# Phase 0: Oracle Batch验证（可选）
# ============================================
if ($Phase0Only -or (-not $SkipPhase0 -and -not $Phase1Only -and -not $Phase2Only -and -not $Phase3Only -and -not $Phase4Only)) {
    Write-Host "========================================="
    Write-Host "Phase 0: Oracle Batch Validation"
    Write-Host "========================================="
    Write-Host "Purpose: Determine upper bound performance (batch mode)"
    Write-Host ""
    
    & ".\scripts\windows\phase0_oracle_batch.ps1" -ConfigFile $ConfigFile
    
    if ($Phase0Only) {
        Write-Host ""
        Write-Host "Phase 0 completed. Exiting." -ForegroundColor Green
        exit 0
    }
    
    Write-Host ""
}

# ============================================
# Phase 1: 长序列内存效率对比（必须）
# ============================================
if ($Phase1Only -or (-not $Phase0Only -and -not $Phase2Only -and -not $Phase3Only -and -not $Phase4Only)) {
    Write-Host "========================================="
    Write-Host "Phase 1: Long Sequence Memory Efficiency"
    Write-Host "========================================="
    Write-Host "Methods: Baseline, H2O, StreamingLLM, Head-Aware"
    Write-Host "Sequence Lengths: $($LONG_SEQUENCE_LENGTHS -join ', ') tokens"
    Write-Host ""
    
    & ".\scripts\windows\phase1_long_sequence.ps1" -ConfigFile $ConfigFile
    
    if ($Phase1Only) {
        Write-Host ""
        Write-Host "Phase 1 completed. Exiting." -ForegroundColor Green
        exit 0
    }
    
    Write-Host ""
}

# ============================================
# Phase 2: 预算影响分析（必须）
# ============================================
if ($Phase2Only -or (-not $Phase0Only -and -not $Phase1Only -and -not $Phase3Only -and -not $Phase4Only)) {
    Write-Host "========================================="
    Write-Host "Phase 2: Budget Impact Analysis"
    Write-Host "========================================="
    Write-Host "Method: Head-Aware"
    Write-Host "Budgets: $($BUDGETS -join ', ') tokens/layer"
    Write-Host ""
    
    & ".\scripts\windows\phase2_budget_analysis.ps1" -ConfigFile $ConfigFile
    
    if ($Phase2Only) {
        Write-Host ""
        Write-Host "Phase 2 completed. Exiting." -ForegroundColor Green
        exit 0
    }
    
    Write-Host ""
}

# ============================================
# Phase 3: 消融实验（必须）
# ============================================
if ($Phase3Only -or (-not $Phase0Only -and -not $Phase1Only -and -not $Phase2Only -and -not $Phase4Only)) {
    Write-Host "========================================="
    Write-Host "Phase 3: Ablation Study"
    Write-Host "========================================="
    Write-Host "Configurations: Baseline, Head-Aware"
    Write-Host "Sequence Length: 5000 tokens"
    Write-Host ""
    
    & ".\scripts\windows\phase3_ablation.ps1" -ConfigFile $ConfigFile
    
    if ($Phase3Only) {
        Write-Host ""
        Write-Host "Phase 3 completed. Exiting." -ForegroundColor Green
        exit 0
    }
    
    Write-Host ""
}

# ============================================
# Phase 4: 结果分析和可视化（推荐）
# ============================================
if ($Phase4Only -or (-not $SkipPhase4 -and -not $Phase0Only -and -not $Phase1Only -and -not $Phase2Only -and -not $Phase3Only)) {
    Write-Host "========================================="
    Write-Host "Phase 4: Results Analysis & Visualization"
    Write-Host "========================================="
    Write-Host ""
    
    & ".\scripts\windows\phase4_analysis.ps1" -ConfigFile $ConfigFile
    
    if ($Phase4Only) {
        Write-Host ""
        Write-Host "Phase 4 completed. Exiting." -ForegroundColor Green
        exit 0
    }
    
    Write-Host ""
}

# ============================================
# 总结
# ============================================
$end_time = Get-Date
$duration = $end_time - $start_time

Write-Host ""
Write-Host "========================================="
Write-Host "All Experiments Completed!"
Write-Host "========================================="
Write-Host ""
Write-Host "Start Time: $start_time"
Write-Host "End Time: $end_time"
Write-Host "Total Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s"
Write-Host ""
Write-Host "Results Directory: $BASE_OUTPUT_DIR"
Write-Host ""
Write-Host "Generated Files:"
Write-Host "  - Long Sequence Results: $BASE_OUTPUT_DIR\long_seq_*"
Write-Host "  - Budget Analysis: $BASE_OUTPUT_DIR\budget_*"
Write-Host "  - Ablation Results: $BASE_OUTPUT_DIR\ablation\*"
Write-Host "  - Analysis Tables: $BASE_OUTPUT_DIR\*_summary.csv"
Write-Host "  - Visualizations: $BASE_OUTPUT_DIR\figures\*"
Write-Host ""
Write-Host "Next Steps:"
Write-Host "1. Review results: $BASE_OUTPUT_DIR"
Write-Host "2. Check analysis files: $BASE_OUTPUT_DIR\*_summary.csv"
Write-Host "3. Review visualizations: $BASE_OUTPUT_DIR\figures"
Write-Host "4. Prepare paper tables from LaTeX files"
Write-Host ""
