# Linux Installation Guide

Complete installation guide for NVFP4-DDIM Optimizer on Linux systems.

## System Requirements

### Supported Distributions
- Ubuntu 20.04 LTS or later
- Debian 11 or later
- Fedora 35 or later
- CentOS 8 or later
- Arch Linux (latest)

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

### Method 1: Automated Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization

# Run installation script
chmod +x install.sh
./install.sh

# Activate environment
source venv/bin/activate
```

The script will:
- Check Python version (3.8+ required)
- Create virtual environment
- Install PyTorch (with CUDA if GPU detected)
- Install all dependencies
- Optionally install development and metrics packages

### Method 2: Manual Installation

#### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

**Fedora:**
```bash
sudo dnf install -y python3 python3-pip git
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip git
```

#### Step 2: Install NVIDIA Drivers (Optional, for GPU support)

**Ubuntu/Debian:**
```bash
# Check if NVIDIA GPU is present
lspci | grep -i nvidia

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot
```

**Fedora:**
```bash
sudo dnf install -y akmod-nvidia
sudo reboot
```

#### Step 3: Clone Repository

```bash
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
```

#### Step 4: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 5: Install PyTorch

**With CUDA (GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Step 6: Install Package

```bash
# Install in development mode
pip install -e .

# Or install from PyPI (when available)
# pip install nvfp4-ddim-optimizer
```

#### Step 7: Install Optional Dependencies

**Development tools:**
```bash
pip install -r requirements-dev.txt
```

**Quality metrics:**
```bash
pip install lpips pytorch-msssim torchmetrics
pip install git+https://github.com/openai/CLIP.git
```

### Method 3: Using Make

```bash
# Install package
make install

# Install with development dependencies
make install-dev
```

## Verification

### Check Installation

```bash
# Activate environment
source venv/bin/activate

# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check package installation
python -c "import nvfp4_ddim_optimizer; print('Package installed successfully!')"
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
make test-cov
```

## Troubleshooting

### Issue: Python version too old

**Error:** `Python 3.8 or higher is required`

**Solution:**
```bash
# Ubuntu/Debian - Install Python 3.10
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Use Python 3.10 explicitly
python3.10 -m venv venv
source venv/bin/activate
```

### Issue: CUDA not detected

**Error:** `torch.cuda.is_available()` returns `False`

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# If driver not found, install it
sudo apt install -y nvidia-driver-535
sudo reboot

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory during installation

**Error:** `MemoryError` or killed during pip install

**Solution:**
```bash
# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Install with no cache
pip install --no-cache-dir -e .
```

### Issue: Permission denied

**Error:** `Permission denied` when running scripts

**Solution:**
```bash
# Make scripts executable
chmod +x install.sh
chmod +x examples/*.py

# Or run with python explicitly
python examples/basic_optimization.py
```

### Issue: Missing system libraries

**Error:** `ImportError: libGL.so.1: cannot open shared object file`

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install -y libgl1-mesa-glx libglib2.0-0

# Fedora
sudo dnf install -y mesa-libGL glib2
```

## Performance Optimization

### Enable CUDA Optimizations

```bash
# Set environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
```

### Use Multiple GPUs

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Run with multi-GPU support
python examples/basic_optimization.py --multi-gpu
```

### Optimize CPU Performance

```bash
# Set number of threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/

# Remove package (if installed globally)
pip uninstall nvfp4-ddim-optimizer

# Remove repository
cd ..
rm -rf diffusion-storage-optimization/
```

## Next Steps

After installation:

1. **Read the documentation**: Check `src/nvfp4_ddim_optimizer/README.md`
2. **Run examples**: Try `examples/basic_optimization.py`
3. **Run benchmarks**: Execute `make benchmark`
4. **Explore presets**: Test different optimization presets

## Support

For issues and questions:
- GitHub Issues: https://github.com/roshan801302/diffusion-storage-optimization/issues
- Documentation: See `docs/` directory
- Examples: See `examples/` directory
