# Windows Installation Guide

Complete installation guide for NVFP4-DDIM Optimizer on Windows.

## System Requirements

### Supported Versions
- Windows 10 (version 1903 or later)
- Windows 11
- Windows Server 2019 or later

### Hardware Requirements

**Minimum:**
- CPU: x86_64 processor
- RAM: 8GB
- Storage: 10GB free space
- GPU: Optional (CPU-only mode supported)

**Recommended:**
- CPU: Modern x86_64 processor (4+ cores)
- RAM: 16GB+
- Storage: 20GB+ free space (SSD recommended)
- GPU: NVIDIA GPU with 6GB+ VRAM and CUDA 11.8+

## Installation Methods

### Method 1: Native Windows Installation (Recommended)

#### Step 1: Install Python

Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/):

```powershell
# Or use winget
winget install Python.Python.3.11
```

Verify installation:
```powershell
python --version
```

#### Step 2: Install Git

Download from [git-scm.com](https://git-scm.com/download/win) or use winget:

```powershell
winget install Git.Git
```

#### Step 3: Clone Repository

```powershell
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
```

#### Step 4: Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### Step 5: Install PyTorch

**With CUDA (GPU):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Step 6: Install Package

```powershell
pip install -e .
```

#### Step 7: Install Optional Dependencies

```powershell
# Development tools
pip install -r requirements-dev.txt

# Quality metrics
pip install lpips pytorch-msssim torchmetrics
pip install git+https://github.com/openai/CLIP.git
```

### Method 2: Using WSL2 (Windows Subsystem for Linux)

WSL2 provides better performance and full Linux compatibility:

#### Step 1: Install WSL2

```powershell
# Run as Administrator
wsl --install
```

Restart your computer.

#### Step 2: Install Ubuntu

```powershell
wsl --install -d Ubuntu
```

#### Step 3: Follow Linux Instructions

Inside WSL2 Ubuntu:
```bash
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
./install.sh
source venv/bin/activate
```

### Method 3: Using Automated PowerShell Script

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run installation script
.\install_windows.ps1
```

## Windows-Specific Configuration

### 1. Enable Long Path Support

Windows has a 260-character path limit. Enable long paths:

```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### 2: Configure Environment Variables

```powershell
# Set environment variables
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

# Make permanent
[System.Environment]::SetEnvironmentVariable('OMP_NUM_THREADS', '4', 'User')
[System.Environment]::SetEnvironmentVariable('MKL_NUM_THREADS', '4', 'User')
```

### 3. Install Visual C++ Redistributable

Some packages require Visual C++ runtime:

Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

Or use winget:
```powershell
winget install Microsoft.VCRedist.2015+.x64
```

## GPU Support on Windows

### Check NVIDIA GPU

```powershell
nvidia-smi
```

### Install CUDA Toolkit (Optional)

Download from: https://developer.nvidia.com/cuda-downloads

Or use winget:
```powershell
winget install NVIDIA.CUDAToolkit
```

### Verify CUDA

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Verification

### Run Setup Verification

```powershell
# Activate environment
.\venv\Scripts\activate

# Run verification
python verify_setup.py
```

Expected output:
```
============================================================
NVFP4-DDIM Optimizer Setup Verification
============================================================
Checking Python version...
  Python 3.x.x
  ✅ Python version OK

Checking platform...
  Platform: Windows
  ✅ Running on Windows

Checking PyTorch...
  PyTorch version: 2.x.x
  CUDA available: True/False
  ✅ PyTorch OK

...

✅ Setup verification PASSED
```

## Troubleshooting

### Issue: Python not found

**Solution:**
```powershell
# Add Python to PATH
$env:Path += ";C:\Users\YourUsername\AppData\Local\Programs\Python\Python311"

# Or reinstall Python with "Add to PATH" checked
```

### Issue: pip not found

**Solution:**
```powershell
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Issue: Permission denied

**Solution:**
```powershell
# Run PowerShell as Administrator
# Or change execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: CUDA not detected

**Solution:**
```powershell
# Check NVIDIA driver
nvidia-smi

# Update driver from: https://www.nvidia.com/Download/index.aspx

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Long path errors

**Solution:**
```powershell
# Enable long paths (run as Administrator)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Restart computer
```

### Issue: SSL certificate errors

**Solution:**
```powershell
# Update certificates
pip install --upgrade certifi

# Or use --trusted-host
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch
```

## Performance Optimization

### Enable Windows Performance Mode

```powershell
# Set power plan to High Performance
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

### Optimize for GPU

```powershell
# Set CUDA environment variables
$env:CUDA_LAUNCH_BLOCKING = "0"
$env:TORCH_CUDA_ARCH_LIST = "7.0;7.5;8.0;8.6"
```

### Use Multiple GPUs

```powershell
# Set visible GPUs
$env:CUDA_VISIBLE_DEVICES = "0,1"
```

## Running Examples

### Basic Usage

```powershell
# Activate environment
.\venv\Scripts\activate

# Run example (when implemented)
python examples\basic_optimization.py --device cuda

# Use fast preset
python examples\basic_optimization.py --preset fast --device cpu
```

### Batch Processing

```powershell
python examples\batch_processing.py `
    --batch-size 4 `
    --device cuda `
    --preset balanced
```

## Windows-Specific Features

### File Paths

Use Windows-style paths or forward slashes:

```python
# Both work
model_path = "C:\\Users\\YourName\\models\\model.pt"
model_path = "C:/Users/YourName/models/model.pt"
```

### PowerShell Integration

```powershell
# Create PowerShell function
function Optimize-Image {
    param($prompt, $output)
    python -c "
from nvfp4_ddim_optimizer import OptimizationPipeline
pipeline = OptimizationPipeline.from_preset('stabilityai/stable-diffusion-2-1-base', 'fast')
pipeline.load_model()
image = pipeline.generate('$prompt')
image.save('$output')
"
}

# Use it
Optimize-Image -prompt "a beautiful landscape" -output "output.png"
```

## Compatibility Notes

### What Works on Windows
- ✅ CPU and GPU inference
- ✅ NVFP4 quantization (87.5% storage reduction)
- ✅ DDIM sampling (4-20× speedup)
- ✅ Batch processing
- ✅ Model save/load
- ✅ Quality metrics
- ✅ Multi-GPU support

### Known Limitations
- ⚠️ Some Linux-specific optimizations not available
- ⚠️ Path length limitations (enable long paths)
- ⚠️ Case-insensitive file system

### Recommended Settings for Windows
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

# Use balanced preset with GPU
pipeline = OptimizationPipeline.from_preset(
    model_id="stabilityai/stable-diffusion-2-1-base",
    preset="balanced",
    device="cuda"  # or "cpu"
)

# Generate
image = pipeline.generate(
    prompt="a beautiful landscape",
    num_inference_steps=50,
    height=512,
    width=512
)
```

## Uninstallation

```powershell
# Deactivate environment
deactivate

# Remove virtual environment
Remove-Item -Recurse -Force venv

# Uninstall package (if installed globally)
pip uninstall nvfp4-ddim-optimizer

# Remove repository
cd ..
Remove-Item -Recurse -Force diffusion-storage-optimization
```

## Support and Resources

### Windows-Specific
- Windows Terminal: https://aka.ms/terminal
- PowerShell Documentation: https://docs.microsoft.com/powershell
- WSL2 Documentation: https://docs.microsoft.com/windows/wsl

### Project Resources
- GitHub: https://github.com/roshan801302/diffusion-storage-optimization/tree/main
- Issues: https://github.com/roshan801302/diffusion-storage-optimization/issues
- Documentation: See `docs/` directory

## Next Steps

After installation:

1. **Verify setup**: `python verify_setup.py`
2. **Read documentation**: `QUICK_START.md`
3. **Run examples**: `python examples\basic_optimization.py`
4. **Optimize for your hardware**: Adjust batch size and step counts

---

**Note**: For best performance on Windows, consider using WSL2 for Linux-like environment or native Windows with CUDA support.
