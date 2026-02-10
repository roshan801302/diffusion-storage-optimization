# Quick Start Guide

## ✅ What's Been Set Up

Your NVFP4-DDIM Optimizer project is now ready for Linux! Here's what's been created:

### 📁 Project Structure
- **Core package**: `src/nvfp4_ddim_optimizer/` with all modules
- **Data models**: Quantization and sampling configurations with validation
- **Test infrastructure**: Pytest with fixtures and Hypothesis strategies
- **Linux tools**: Installation script, Makefile, and comprehensive docs

### 🐧 Linux-Specific Features
- Automated installation script with GPU detection
- Makefile for common commands
- Proper file permissions for executables
- Distribution-specific installation instructions

## 🚀 Installation (Choose One Method)

### Method 1: Automated (Recommended)

```bash
# Run the installation script
./install.sh

# Activate environment
source venv/bin/activate

# Verify installation
python verify_setup.py
```

### Method 2: Manual

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (with CUDA if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install package
pip install -e .

# Install dev dependencies (optional)
pip install -r requirements-dev.txt

# Verify installation
python verify_setup.py
```

### Method 3: Using Make

```bash
# Create venv first
python3 -m venv venv
source venv/bin/activate

# Install with Make
make install-dev

# Verify
python verify_setup.py
```

## 📋 What Works Right Now

### ✅ Implemented
1. **Configuration Classes**
   - `QuantizationConfig` with validation
   - `SamplingConfig` with validation
   - Automatic parameter validation with helpful error messages

2. **Data Models**
   - `QuantizedTensor` with dequantization support
   - `QuantizedModel` with save/load functionality
   - Compression ratio calculation

3. **Testing Infrastructure**
   - Pytest configuration
   - Hypothesis strategies for property-based testing
   - Test fixtures for common objects

4. **Development Tools**
   - Makefile with common commands
   - Installation script for Linux
   - Verification script
   - Comprehensive documentation

### ⏳ To Be Implemented (Next Steps)
- NVFP4 quantization engine (Task 2)
- Calibration engine (Task 4)
- DDIM scheduler (Task 7-8)
- Quality metrics (Task 9)
- Optimization pipeline (Task 11-13)
- Examples and benchmarks (Task 19-20)

## 🎯 Next Steps

### 1. Install Dependencies

```bash
# Activate environment
source venv/bin/activate

# Run installation
./install.sh
# OR
make install-dev
```

### 2. Verify Setup

```bash
python verify_setup.py
```

You should see:
```
✅ Setup verification PASSED

You're ready to start development!
```

### 3. Start Development

Choose one of these paths:

**Option A: Continue with Task 2 (Quantization Core)**
```bash
# Read the task
cat ~/.kiro/specs/nvfp4-ddim-optimizer/tasks.md | grep -A 20 "Task 2"

# Start implementing
# Edit: src/nvfp4_ddim_optimizer/quantization/quantizer.py
```

**Option B: Run Tests**
```bash
# Run existing tests
make test

# Run with coverage
make test-cov
```

**Option C: Explore the Code**
```bash
# Check what's implemented
python -c "from nvfp4_ddim_optimizer import *; help(QuantizationConfig)"
```

## 📚 Documentation

- **`SETUP_COMPLETE.md`**: Detailed overview of what's been set up
- **`INSTALL_LINUX.md`**: Complete Linux installation guide with troubleshooting
- **`src/nvfp4_ddim_optimizer/README.md`**: Package documentation
- **`~/.kiro/specs/nvfp4-ddim-optimizer/`**: Full specification
  - `requirements.md`: 14 requirements with acceptance criteria
  - `design.md`: Architecture and component interfaces
  - `tasks.md`: 22 implementation tasks

## 🔧 Common Commands

```bash
# Installation
make install          # Install package
make install-dev      # Install with dev dependencies

# Testing
make test            # Run tests
make test-cov        # Run tests with coverage

# Code Quality
make format          # Format code with black
make lint            # Run linting checks

# Cleanup
make clean           # Remove build artifacts

# Help
make help            # Show all commands
```

## 💡 Usage Example

Once installed, you can use the configurations:

```python
from nvfp4_ddim_optimizer import QuantizationConfig, SamplingConfig

# Create quantization config
quant_config = QuantizationConfig(
    strategy="per_channel",
    calibration_method="minmax",
    num_calibration_samples=100
)

# Create sampling config
sampling_config = SamplingConfig(
    num_inference_steps=50,
    schedule_type="uniform",
    eta=0.0
)

print(f"Quantization: {quant_config.strategy}")
print(f"Sampling: {sampling_config.num_inference_steps} steps")
```

## 🐛 Troubleshooting

### Issue: Python version too old
```bash
# Install Python 3.10+
sudo apt install python3.10 python3.10-venv
python3.10 -m venv venv
```

### Issue: CUDA not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Install driver if needed
sudo apt install nvidia-driver-535
sudo reboot
```

### Issue: Permission denied
```bash
# Make scripts executable
chmod +x install.sh verify_setup.py
```

See `INSTALL_LINUX.md` for more troubleshooting.

## 📊 Project Status

| Task | Status | Description |
|------|--------|-------------|
| Task 1 | ✅ Complete | Project structure and data models |
| Task 2 | 📝 Ready | NVFP4 quantization core |
| Task 3 | ⏳ Pending | Quantization parameter computation |
| Task 4 | ⏳ Pending | Calibration engine |
| Tasks 5-22 | ⏳ Pending | See tasks.md for details |

## 🎓 Learning Resources

- **Spec files**: Read the requirements and design documents
- **Code**: Explore `src/nvfp4_ddim_optimizer/quantization/data_models.py`
- **Tests**: Check `tests/conftest.py` for test patterns
- **Examples**: Will be created in Task 19

## 🤝 Ready to Code?

1. ✅ Install dependencies: `./install.sh`
2. ✅ Verify setup: `python verify_setup.py`
3. ✅ Read the spec: `~/.kiro/specs/nvfp4-ddim-optimizer/`
4. 🚀 Start Task 2: Implement NVFP4 quantization core

---

**Questions?** Check the documentation or run `make help` for available commands.
