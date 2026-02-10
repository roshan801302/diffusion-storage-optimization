# Platform Quick Reference

Quick reference for installing and running NVFP4-DDIM Optimizer on different platforms.

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

## 📱 HarmonyOS

### Installation
```bash
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization
./install_harmonyos.sh
source venv/bin/activate
```

### Usage
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

pipeline = OptimizationPipeline.from_preset(
    "stabilityai/stable-diffusion-2-1-base",
    preset="fast",  # Recommended for HarmonyOS
    device="cpu"
)

image = pipeline.generate(
    "a beautiful landscape",
    num_inference_steps=20,
    height=512,
    width=512
)
```

### Performance
- Memory: 0.43 GB (87.5% reduction)
- Speed: 2-4× faster (CPU-only)
- Quality: FID +7.9%

### Documentation
- `INSTALL_HARMONYOS.md`

---

## 📊 Feature Comparison

| Feature | Linux | HarmonyOS |
|---------|-------|-----------|
| GPU Support | ✅ CUDA | ⚠️ Limited |
| CPU Mode | ✅ | ✅ |
| Quantization | ✅ | ✅ |
| DDIM Sampling | ✅ | ✅ |
| Performance | Excellent | Good |
| Status | Stable | Experimental |

---

## 🚀 Quick Commands

### Installation
```bash
# Linux
./install.sh

# HarmonyOS
./install_harmonyos.sh
```

### Verification
```bash
python verify_setup.py
```

### Testing
```bash
make test
```

### Examples
```bash
# Linux with GPU
python examples/basic_optimization.py --device cuda --preset balanced

# HarmonyOS CPU
python examples/basic_optimization.py --device cpu --preset fast
```

---

## 📚 Documentation

- **Quick Start**: `QUICK_START.md`
- **Linux Guide**: `INSTALL_LINUX.md`
- **HarmonyOS Guide**: `INSTALL_HARMONYOS.md`
- **Platform Support**: `PLATFORM_SUPPORT.md`
- **Updates**: `UPDATES_SUMMARY.md`

---

## 🔗 Links

- **Repository**: https://github.com/roshan801302/diffusion-storage-optimization/tree/main
- **Issues**: https://github.com/roshan801302/diffusion-storage-optimization/issues
- **Author**: rr
- **License**: MIT
