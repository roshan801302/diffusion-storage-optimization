# Installation script for NVFP4-DDIM Optimizer on Windows
# Run as: .\install_windows.ps1

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "NVFP4-DDIM Optimizer - Windows Installation" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
    
    # Check if Python >= 3.8
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($matches) {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Host "Error: Python 3.8 or higher is required" -ForegroundColor Red
            Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
            exit 1
        }
    }
} catch {
    Write-Host "Error: Python not found" -ForegroundColor Red
    Write-Host "Please install Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host ""
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Check for NVIDIA GPU
Write-Host ""
Write-Host "Checking for NVIDIA GPU..." -ForegroundColor Yellow
$hasNvidiaGPU = $false
try {
    $nvidiaCheck = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "NVIDIA GPU detected!" -ForegroundColor Green
        $hasNvidiaGPU = $true
    }
} catch {
    Write-Host "No NVIDIA GPU detected" -ForegroundColor Yellow
}

# Install PyTorch
Write-Host ""
if ($hasNvidiaGPU) {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "Installing PyTorch (CPU-only)..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# Install package
Write-Host ""
Write-Host "Installing NVFP4-DDIM Optimizer..." -ForegroundColor Yellow
pip install -e .

# Ask about development dependencies
Write-Host ""
$devDeps = Read-Host "Install development dependencies? (y/n)"
if ($devDeps -eq 'y' -or $devDeps -eq 'Y') {
    Write-Host "Installing development dependencies..." -ForegroundColor Yellow
    pip install -r requirements-dev.txt
}

# Ask about quality metrics
Write-Host ""
$metricsDeps = Read-Host "Install quality metrics dependencies (FID, LPIPS, CLIP)? (y/n)"
if ($metricsDeps -eq 'y' -or $metricsDeps -eq 'Y') {
    Write-Host "Installing quality metrics dependencies..." -ForegroundColor Yellow
    pip install lpips pytorch-msssim torchmetrics
    Write-Host "Installing CLIP (may take time)..." -ForegroundColor Yellow
    pip install git+https://github.com/openai/CLIP.git
}

# Set environment variables
Write-Host ""
Write-Host "Configuring environment variables..." -ForegroundColor Yellow
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

# Add to activation script
$activateScript = ".\venv\Scripts\Activate.ps1"
$envVars = @"

# NVFP4-DDIM Optimizer optimizations
`$env:OMP_NUM_THREADS = "4"
`$env:MKL_NUM_THREADS = "4"
Write-Host "NVFP4-DDIM Optimizer environment activated" -ForegroundColor Green
"@

Add-Content -Path $activateScript -Value $envVars

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
python verify_setup.py

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To run tests:" -ForegroundColor Yellow
Write-Host "  pytest tests\" -ForegroundColor White
Write-Host ""
Write-Host "To run examples:" -ForegroundColor Yellow
Write-Host "  python examples\basic_optimization.py" -ForegroundColor White
Write-Host ""
Write-Host "For Windows-specific help, see:" -ForegroundColor Yellow
Write-Host "  INSTALL_WINDOWS.md" -ForegroundColor White
Write-Host ""
