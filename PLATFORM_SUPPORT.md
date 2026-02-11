# Platform Support

NVFP4-DDIM Optimizer supports multiple platforms with optimizations for each.

## Supported Platforms

### ✅ Linux (Primary Platform)
- **Status**: Fully Supported
- **Distributions**: Ubuntu 20.04+, Debian 11+, Fedora 35+, CentOS 8+, Arch Linux
- **GPU Support**: NVIDIA CUDA 11.8+
- **Installation**: `./install.sh`
- **Documentation**: `INSTALL_LINUX.md`

**Features:**
- Full GPU acceleration with CUDA
- Optimized memory management
- Multi-GPU support
- All quality metrics (FID, LPIPS, CLIP)

### ✅ Windows (Fully Supported)
- **Status**: Fully Supported
- **Versions**: Windows 10 (1903+), Windows 11, Windows Server 2019+
- **GPU Support**: NVIDIA CUDA 11.8+
- **Installation**: `.\install_windows.ps1` or WSL2
- **Documentation**: `INSTALL_WINDOWS.md`

**Features:**
- Native Windows support
- GPU acceleration with CUDA
- WSL2 for Linux compatibility
- PowerShell integration
- All quality metrics

### ✅ OpenKylin (Fully Supported)
- **Status**: Fully Supported
- **Versions**: OpenKylin 1.0+, OpenKylin 2.0 (recommended)
- **Architectures**: x86_64, ARM64
- **GPU Support**: NVIDIA CUDA 11.8+ (x86_64), CPU-only (ARM64)
- **Installation**: `./install.sh`
- **Documentation**: `INSTALL_OPENKYLIN.md`

**Features:**
- Full support for x86_64 and ARM64
- GPU acceleration on x86_64 with NVIDIA
- Optimized for Chinese users
- UKUI desktop integration
- All quality metrics

### ⚠️ macOS (Community Support)
- **Status**: Community Support
- **Versions**: macOS 11.0+ (Big Sur and later)
- **GPU Support**: Metal (experimental)
- **Installation**: Similar to Linux
- **Documentation**: Use `INSTALL_LINUX.md` as reference

**Features:**
- CPU inference
- Metal GPU support (experimental)
- All core features

**Limitations:**
- Metal support limited
- Performance varies by hardware

## Installation by Platform

### Linux

```bash
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
./install.sh
source venv/bin/activate
```

### Windows

**PowerShell:**
```powershell
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
.\install_windows.ps1
.\venv\Scripts\Activate.ps1
```

**WSL2:**
```bash
wsl --install
# Then follow Linux instructions
```

### OpenKylin

```bash
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
./install.sh
source venv/bin/activate
```

### macOS

```bash
# Similar to Linux
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
./install.sh
source venv/bin/activate
```

## Feature Comparison

| Feature | Linux | Windows | OpenKylin | macOS |
|---------|-------|---------|-----------|-------|
| NVFP4 Quantization | ✅ | ✅ | ✅ | ✅ |
| DDIM Sampling | ✅ | ✅ | ✅ | ✅ |
| GPU Acceleration | ✅ CUDA | ✅ CUDA | ✅ CUDA (x86_64) | ⚠️ Metal |
| CPU-only Mode | ✅ | ✅ | ✅ | ✅ |
| Quality Metrics | ✅ All | ✅ All | ✅ All | ✅ All |
| Batch Processing | ✅ | ✅ | ✅ | ✅ |
| Multi-GPU | ✅ | ✅ | ✅ (x86_64) | ❌ |
| ARM64 Support | ✅ | ❌ | ✅ | ✅ (M1/M2) |
| Performance | Excellent | Excellent | Excellent | Good |
| Memory Efficiency | Excellent | Excellent | Excellent | Good |
| Status | Stable | Stable | Stable | Community |

## Performance Benchmarks

### Linux (NVIDIA RTX 3090)
```
Configuration: Balanced preset
Memory: 0.43 GB (87.5% reduction)
Time: 1.06s per image
Speedup: 8.0×
Quality: FID +3.9%
```

### HarmonyOS (CPU-only)
```
Configuration: Fast preset
Memory: 0.43 GB (87.5% reduction)
Time: ~8-12s per image (device-dependent)
Speedup: 2-4× (vs unoptimized CPU)
Quality: FID +7.9%
```

## Recommended Configurations

### Linux with GPU
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

pipeline = OptimizationPipeline.from_preset(
    model_id="stabilityai/stable-diffusion-2-1-base",
    preset="balanced",  # or "quality"
    device="cuda"
)
```

### HarmonyOS (CPU)
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

pipeline = OptimizationPipeline.from_preset(
    model_id="stabilityai/stable-diffusion-2-1-base",
    preset="fast",  # Optimized for mobile
    device="cpu"
)

# Use lower resolution for better performance
image = pipeline.generate(
    prompt="a beautiful landscape",
    num_inference_steps=20,
    height=512,
    width=512
)
```

### Linux CPU-only
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

pipeline = OptimizationPipeline.from_preset(
    model_id="stabilityai/stable-diffusion-2-1-base",
    preset="balanced",
    device="cpu"
)
```

## Platform-Specific Optimizations

### Linux
- CUDA kernel optimizations
- Multi-GPU data parallelism
- Optimized memory allocation
- xFormers attention (if available)

### HarmonyOS
- Aggressive memory management
- CPU thread optimization
- Reduced batch sizes
- Optimized model caching
- Background task support

## Troubleshooting by Platform

### Linux Issues

**GPU not detected:**
```bash
nvidia-smi
sudo apt install nvidia-driver-535
```

**Out of memory:**
```bash
# Reduce batch size or use CPU offloading
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### HarmonyOS Issues

**Python not found:**
```bash
# Install from App Gallery or use:
apt-get install python3 python3-pip
```

**Slow performance:**
```bash
# Use fast preset and lower resolution
export OMP_NUM_THREADS=4
```

**Memory issues:**
```bash
# Enable memory optimizations
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## Testing by Platform

### Linux
```bash
# Full test suite
make test

# With GPU
pytest tests/ -v --device cuda
```

### HarmonyOS
```bash
# CPU-only tests
pytest tests/ -v --device cpu

# Skip GPU tests
pytest tests/ -v -m "not gpu"
```

## Contributing

When contributing, please test on:
1. Linux with GPU (primary)
2. Linux CPU-only (secondary)
3. HarmonyOS (if available)

## Support

- **Linux**: Full support via GitHub Issues
- **HarmonyOS**: Experimental support, community-driven
- **Other platforms**: Community support only

## Future Platform Support

Planned:
- ✅ Linux (Complete)
- ✅ HarmonyOS (Experimental)
- 🔄 Windows native (Planned)
- 🔄 macOS (Planned)
- ❓ Android (Under consideration)

## Resources

- **GitHub**: https://github.com/roshan801302/diffusion-storage-optimization/tree/main
- **Issues**: https://github.com/roshan801302/diffusion-storage-optimization/issues
- **Linux Guide**: `INSTALL_LINUX.md`
- **HarmonyOS Guide**: `INSTALL_HARMONYOS.md`
- **Quick Start**: `QUICK_START.md`

---

**Author**: rr  
**License**: MIT  
**Repository**: https://github.com/roshan801302/diffusion-storage-optimization/tree/main
