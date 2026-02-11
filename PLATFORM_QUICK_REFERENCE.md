# Platform Quick Reference

Quick reference for installing and running NVFP4-DDIM Optimizer on all supported platforms.

## 🐧 Linux

### Installation
```bash
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
./install.sh
source venv/bin/activate
```

### Usage
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

pipeline = OptimizationPipeline.from_preset(
    "stabilityai/stable-diffusion-2-1-base",
    preset="balanced",
    device="cuda"  # or "cpu"
)
```

### Performance
- Memory: 0.43 GB (87.5% reduction)
- Speed: 8× faster (with GPU)
- Quality: FID +3.9%

### Documentation
- `INSTALL_LINUX.md`

---

## 🪟 Windows

### Installation
```powershell
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
.\install_windows.ps1
.\venv\Scripts\Activate.ps1
```

### Usage
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

pipeline = OptimizationPipeline.from_preset(
    "stabilityai/stable-diffusion-2-1-base",
    preset="balanced",
    device="cuda"  # or "cpu"
)
```

### Performance
- Memory: 0.43 GB (87.5% reduction)
- Speed: 8× faster (with GPU)
- Quality: FID +3.9%

### Documentation
- `INSTALL_WINDOWS.md`

---

## 🐉 OpenKylin

### Installation
```bash
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
./install.sh
source venv/bin/activate
```

### Usage
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

pipeline = OptimizationPipeline.from_preset(
    "stabilityai/stable-diffusion-2-1-base",
    preset="balanced",
    device="cuda"  # or "cpu"
)
```

### Performance
- Memory: 0.43 GB (87.5% reduction)
- Speed: 8× faster (with GPU on x86_64)
- Quality: FID +3.9%

### Documentation
- `INSTALL_OPENKYLIN.md`

---

## 🍎 macOS

### Installation
```bash
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
./install.sh
source venv/bin/activate
```

### Usage
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

pipeline = OptimizationPipeline.from_preset(
    "stabilityai/stable-diffusion-2-1-base",
    preset="balanced",
    device="cpu"  # or "mps" for Metal
)
```

### Performance
- Memory: 0.43 GB (87.5% reduction)
- Speed: 4-6× faster (CPU/Metal)
- Quality: FID +3.9%

### Documentation
- Use `INSTALL_LINUX.md` as reference

---

## 📊 Feature Comparison

| Feature | Linux | Windows | OpenKylin | macOS |
|---------|-------|---------|-----------|-------|
| GPU Support | ✅ CUDA | ✅ CUDA | ✅ CUDA | ⚠️ Metal |
| CPU Mode | ✅ | ✅ | ✅ | ✅ |
| Quantization | ✅ | ✅ | ✅ | ✅ |
| DDIM Sampling | ✅ | ✅ | ✅ | ✅ |
| Performance | Excellent | Excellent | Excellent | Good |
| Status | Stable | Stable | Stable | Community |

---

## 🚀 Quick Commands

### Installation
```bash
# Linux / macOS / OpenKylin
./install.sh

# Windows
.\install_windows.ps1
```

### Verification
```bash
# All platforms
python verify_setup.py
```

### Testing
```bash
# Linux / macOS / OpenKylin
make test

# Windows
pytest tests\
```

### Examples
```bash
# Linux / Windows / OpenKylin with GPU
python examples/basic_optimization.py --device cuda --preset balanced

# macOS (CPU or Metal)
python examples/basic_optimization.py --device mps --preset balanced
```

---

## 📚 Documentation

- **Quick Start**: `QUICK_START.md`
- **Linux Guide**: `INSTALL_LINUX.md`
- **Windows Guide**: `INSTALL_WINDOWS.md`
- **OpenKylin Guide**: `INSTALL_OPENKYLIN.md`
- **Platform Support**: `PLATFORM_SUPPORT.md`

---

## 🔗 Links

- **Repository**: https://github.com/roshan801302/diffusion-storage-optimization/tree/main
- **Issues**: https://github.com/roshan801302/diffusion-storage-optimization/issues
- **Author**: rr
- **License**: MIT
