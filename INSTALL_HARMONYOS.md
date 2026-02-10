# HarmonyOS Installation Guide

Complete installation guide for NVFP4-DDIM Optimizer on HarmonyOS NEXT.

## System Requirements

### Supported Versions
- HarmonyOS NEXT 5.0 or later
- HarmonyOS 4.0 or later (with Linux compatibility layer)

### Hardware Requirements

**Minimum:**
- CPU: ARM64 or x86_64 processor
- RAM: 8GB
- Storage: 10GB free space
- GPU: Optional (CPU-only mode supported)

**Recommended:**
- CPU: Modern ARM64/x86_64 processor (4+ cores)
- RAM: 16GB+
- Storage: 20GB+ free space (SSD recommended)
- GPU: Compatible GPU with compute support

## Installation Methods

### Method 1: Using Python Environment (Recommended)

HarmonyOS NEXT includes Python support through its Linux compatibility layer.

```bash
# Check Python availability
python3 --version

# If Python not available, install from HarmonyOS App Gallery
# Search for "Python" in App Gallery and install

# Clone repository
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-storage-optimization

# Create virtual environment
python3 -m venv venv

# Activate environment (HarmonyOS uses bash-compatible shell)
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for HarmonyOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install package
pip install -e .

# Verify installation
python verify_setup.py
```

### Method 2: Using HarmonyOS DevEco Studio

If you're developing within DevEco Studio:

1. **Open DevEco Studio**
2. **Import Project**: File → Open → Select `diffusion-storage-optimization`
3. **Configure Python Interpreter**:
   - File → Settings → Project → Python Interpreter
   - Add new interpreter or use system Python
4. **Install Dependencies**:
   - Open Terminal in DevEco Studio
   - Run: `pip install -e .`

### Method 3: Using Automated Script

```bash
# Make script executable
chmod +x install.sh

# Run installation
./install.sh

# Activate environment
source venv/bin/activate
```

## HarmonyOS-Specific Configuration

### 1. Enable Linux Compatibility Layer

HarmonyOS NEXT includes a Linux compatibility layer for running Linux applications:

```bash
# Check if Linux compatibility is enabled
uname -a

# If not enabled, go to:
# Settings → System → Developer Options → Enable Linux Compatibility
```

### 2. Configure Python Path

```bash
# Add Python to PATH (if needed)
export PATH="$HOME/.local/bin:$PATH"

# Add to shell profile for persistence
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Install System Dependencies

HarmonyOS uses its own package manager. Install required libraries:

```bash
# Using HarmonyOS package manager (if available)
hpm install python3-dev
hpm install git

# Or use Linux compatibility layer
apt-get update
apt-get install -y python3-dev python3-pip git
```

## GPU Support on HarmonyOS

### Check GPU Availability

```bash
# Check for compute-capable GPU
python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

### CPU-Only Mode (Recommended for HarmonyOS)

For most HarmonyOS devices, CPU-only mode is recommended:

```bash
# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify CPU mode
python3 -c "import torch; print(f'Device: {torch.device(\"cpu\")}')"
```

## Verification

### Run Setup Verification

```bash
# Activate environment
source venv/bin/activate

# Run verification script
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
  Platform: Linux (HarmonyOS)
  ✅ Running on compatible platform

Checking PyTorch...
  PyTorch version: 2.x.x
  CUDA available: False (CPU mode)
  ✅ PyTorch OK

...

✅ Setup verification PASSED
```

## HarmonyOS-Specific Features

### 1. Memory Management

HarmonyOS has aggressive memory management. Configure Python to work within limits:

```bash
# Set memory limits
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=4
```

### 2. Background Execution

For long-running tasks, use HarmonyOS background task API:

```python
# In your Python script
import os
os.nice(10)  # Lower priority for background tasks
```

### 3. File System Access

HarmonyOS has sandboxed file system. Use appropriate directories:

```bash
# Use user data directory
export DATA_DIR="$HOME/Documents/diffusion-models"
mkdir -p $DATA_DIR

# Configure in Python
import os
os.environ['MODEL_CACHE_DIR'] = os.path.expanduser('~/Documents/diffusion-models')
```

## Troubleshooting

### Issue: Python not found

**Solution:**
```bash
# Install Python from HarmonyOS App Gallery
# Or use Linux compatibility layer
apt-get install python3 python3-pip
```

### Issue: Permission denied

**Solution:**
```bash
# Grant execute permissions
chmod +x install.sh verify_setup.py

# Or run with python explicitly
python3 verify_setup.py
```

### Issue: Out of memory

**Solution:**
```bash
# Reduce batch size and model size
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Use CPU-only mode
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Git not available

**Solution:**
```bash
# Install git through HarmonyOS package manager
hpm install git

# Or download ZIP from GitHub
# https://github.com/roshan801302/diffusion-storage-optimization/archive/refs/heads/main.zip
```

### Issue: Slow performance

**Solution:**
```bash
# Optimize for HarmonyOS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Use smaller models and lower step counts
# Edit config to use "fast" preset
```

## Performance Optimization for HarmonyOS

### 1. CPU Optimization

```bash
# Set optimal thread count
export OMP_NUM_THREADS=$(nproc)

# Enable CPU optimizations
export ATEN_CPU_CAPABILITY=default
```

### 2. Memory Optimization

```python
# In your Python code
import torch
torch.set_num_threads(4)  # Adjust based on device

# Use smaller batch sizes
batch_size = 1  # For HarmonyOS devices
```

### 3. Storage Optimization

```bash
# Use quantized models (already implemented)
# Models are 87.5% smaller with NVFP4 quantization

# Clear cache regularly
rm -rf ~/.cache/huggingface/
```

## Running Examples

### Basic Usage

```bash
# Activate environment
source venv/bin/activate

# Run basic example (when implemented)
python examples/basic_optimization.py --device cpu

# Use fast preset for HarmonyOS
python examples/basic_optimization.py --preset fast --device cpu
```

### Batch Processing

```bash
# Process multiple images with low memory
python examples/batch_processing.py \
    --batch-size 1 \
    --device cpu \
    --preset fast
```

## HarmonyOS App Integration

### Creating HarmonyOS App

To integrate into a HarmonyOS app:

1. **Create HAP (HarmonyOS Ability Package)**
2. **Include Python runtime**
3. **Bundle the optimizer package**
4. **Use ArkTS to call Python code**

Example ArkTS integration:

```typescript
import { PythonBridge } from '@ohos.pythonbridge';

// Initialize optimizer
const optimizer = PythonBridge.import('nvfp4_ddim_optimizer');

// Run optimization
const result = optimizer.OptimizationPipeline.from_preset(
  'stabilityai/stable-diffusion-2-1-base',
  'fast'
);
```

## Compatibility Notes

### What Works on HarmonyOS
- ✅ CPU-only inference
- ✅ NVFP4 quantization (87.5% storage reduction)
- ✅ DDIM sampling (4-20× speedup)
- ✅ Batch processing
- ✅ Model save/load
- ✅ Quality metrics

### What's Limited
- ⚠️ GPU acceleration (depends on device)
- ⚠️ Large models (memory constraints)
- ⚠️ Real-time generation (CPU performance)

### Recommended Settings for HarmonyOS
```python
from nvfp4_ddim_optimizer import OptimizationPipeline

# Use fast preset with CPU
pipeline = OptimizationPipeline.from_preset(
    model_id="stabilityai/stable-diffusion-2-1-base",
    preset="fast",  # Optimized for mobile/embedded
    device="cpu"
)

# Generate with low memory settings
image = pipeline.generate(
    prompt="a beautiful landscape",
    num_inference_steps=20,  # Fewer steps for speed
    height=512,  # Lower resolution
    width=512
)
```

## Support and Resources

### HarmonyOS-Specific
- HarmonyOS Developer Forum: https://developer.huawei.com/consumer/en/forum/
- DevEco Studio: https://developer.harmonyos.com/en/develop/deveco-studio

### Project Resources
- GitHub: https://github.com/roshan801302/diffusion-storage-optimization/tree/main
- Issues: https://github.com/roshan801302/diffusion-storage-optimization/issues
- Documentation: See `docs/` directory

## Next Steps

After installation:

1. **Verify setup**: `python verify_setup.py`
2. **Read documentation**: `QUICK_START.md`
3. **Run examples**: `python examples/basic_optimization.py --device cpu`
4. **Optimize for your device**: Adjust batch size and step counts

---

**Note**: HarmonyOS support is experimental. Performance may vary by device. For best results, use CPU-only mode with the "fast" preset.
