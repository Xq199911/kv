# 模型下载脚本 - Windows版本
# 使用国内镜像源

Write-Host "========================================="
Write-Host "Model Download Script (Windows)"
Write-Host "========================================="
Write-Host ""

# 检查Python环境
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion"
} catch {
    Write-Host "❌ Python not found!"
    Write-Host "Please install Python 3.8+ and add it to PATH"
    exit 1
}

# 检查Python版本
$pythonMajor = python -c "import sys; print(sys.version_info.major)" 2>&1
$pythonMinor = python -c "import sys; print(sys.version_info.minor)" 2>&1
$pythonVersion = "$pythonMajor.$pythonMinor"
Write-Host "Python version: $pythonVersion"

# 安装huggingface_hub（必须）
Write-Host "Checking huggingface_hub installation..."
try {
    python -c "import huggingface_hub" 2>&1 | Out-Null
    Write-Host "✅ huggingface_hub already installed"
} catch {
    Write-Host "Installing huggingface_hub..."
    python -m pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
}

# 安装ModelScope（可选，需要Python 3.9+）
Write-Host "Checking ModelScope installation..."
if ([int]$pythonMajor -eq 3 -and [int]$pythonMinor -ge 9) {
    try {
        python -c "import modelscope" 2>&1 | Out-Null
        Write-Host "✅ ModelScope already installed"
    } catch {
        Write-Host "Installing ModelScope (Python 3.9+ required)..."
        python -m pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
    }
} else {
    Write-Host "⚠️  ModelScope requires Python 3.9+, will use HuggingFace mirror instead"
}

# 模型列表
$models = @(
    "Qwen2.5-3B-Instruct",
    "Llama3-8B-Instruct",
    "Gemma2-9B-Instruct"
)

Write-Host ""
Write-Host "Available models:"
for ($i = 0; $i -lt $models.Length; $i++) {
    Write-Host "  $($i+1). $($models[$i])"
}
Write-Host "  4. All models"
Write-Host ""

$choice = Read-Host "Select model to download (1-4)"

switch ($choice) {
    "1" { $MODEL = "Qwen2.5-3B-Instruct" }
    "2" { $MODEL = "Llama3-8B-Instruct" }
    "3" { $MODEL = "Gemma2-9B-Instruct" }
    "4" { $MODEL = "all" }
    default {
        Write-Host "Invalid choice"
        exit 1
    }
}

# 运行下载脚本
Write-Host ""
Write-Host "Starting download..."

# 根据Python版本选择下载脚本
if ([int]$pythonMajor -eq 3 -and [int]$pythonMinor -lt 9) {
    # Python 3.8使用专用脚本
    if (Test-Path "download_models_python38.py") {
        python download_models_python38.py --model $MODEL
    } else {
        python download_models_china.py --model $MODEL
    }
} else {
    # Python 3.9+使用标准脚本
    python download_models_china.py --model $MODEL --use-modelscope
}

Write-Host ""
Write-Host "========================================="
Write-Host "Download completed!"
Write-Host "========================================="
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Verify models: python check_model_integrity.py .\models\<model_name>"
Write-Host "2. Run experiments: .\scripts\windows\run_a_level_experiments.ps1"
Write-Host ""

