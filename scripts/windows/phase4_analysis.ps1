# Phase 4: 结果分析和可视化
# 分析实验结果，生成表格和图表

param(
    [string]$ConfigFile = ".\scripts\windows\config.ps1",
    [string]$ResultsDir = ""  # 如果为空，使用配置文件中的默认值
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
if ($ResultsDir -eq "") {
    $ResultsDir = $BASE_OUTPUT_DIR
}

Write-Host "========================================="
Write-Host "Phase 4: Results Analysis & Visualization"
Write-Host "========================================="
Write-Host "Results Directory: $ResultsDir"
Write-Host ""

# 检查结果目录是否存在
if (-not (Test-Path $ResultsDir)) {
    Write-Host "ERROR: Results directory not found: $ResultsDir" -ForegroundColor Red
    Write-Host "Please run Phase 1-3 first!" -ForegroundColor Yellow
    exit 1
}

# 1. 分析长序列实验结果
Write-Host "[1/3] Analyzing long sequence results..."
$long_seq_dir = "$ResultsDir\long_seq_10000"
if (Test-Path $long_seq_dir) {
    python analyze_experiment_results.py `
        --output_dir $long_seq_dir `
        --detailed `
        --save_csv "$ResultsDir\long_seq_10000_summary.csv" `
        --save_json "$ResultsDir\long_seq_10000_summary.json" `
        --save_latex "$ResultsDir\long_seq_10000_table.tex"
    
    Write-Host "  [OK] Long sequence analysis saved" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Long sequence results not found: $long_seq_dir" -ForegroundColor Yellow
}

Write-Host ""

# 2. 分析消融实验结果
Write-Host "[2/3] Analyzing ablation results..."
$ablation_dir = "$ResultsDir\ablation"
if (Test-Path $ablation_dir) {
    python analyze_experiment_results.py `
        --output_dir $ablation_dir `
        --detailed `
        --save_csv "$ResultsDir\ablation_summary.csv" `
        --save_json "$ResultsDir\ablation_summary.json" `
        --save_latex "$ResultsDir\ablation_table.tex"
    
    Write-Host "  [OK] Ablation analysis saved" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Ablation results not found: $ablation_dir" -ForegroundColor Yellow
}

Write-Host ""

# 3. 生成可视化图表
Write-Host "[3/3] Generating visualizations..."
$figures_dir = "$ResultsDir\figures"
New-Item -ItemType Directory -Force -Path $figures_dir | Out-Null

if (Test-Path $long_seq_dir) {
    python visualize_results.py `
        --results_dir $long_seq_dir `
        --output_dir $figures_dir `
        --include_budget
    
    Write-Host "  [OK] Visualizations saved to: $figures_dir" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Cannot generate visualizations: results not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[OK] Phase 4 completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Generated Files:"
Write-Host "  - CSV: $ResultsDir\*_summary.csv"
Write-Host "  - JSON: $ResultsDir\*_summary.json"
Write-Host "  - LaTeX: $ResultsDir\*_table.tex"
Write-Host "  - Figures: $figures_dir\*"
Write-Host ""

