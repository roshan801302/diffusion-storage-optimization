#!/bin/bash
# Installation script for NVFP4-DDIM Optimizer on HarmonyOS

set -e  # Exit on error

echo "========================================="
echo "NVFP4-DDIM Optimizer - HarmonyOS Installation"
echo "========================================="
echo ""

# Detect platform
echo "Detecting platform..."
platform=$(uname -s)
echo "Platform: $platform"

if [[ "$platform" != "Linux" ]]; then
    echo "Warning: This script is optimized for HarmonyOS (Linux-based)"
    echo "Detected platform: $platform"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo ""
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    echo "Please install Python from HarmonyOS App Gallery or use:"
    echo "  apt-get install python3 python3-pip"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python >= 3.8
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU-only for HarmonyOS)
echo ""
echo "Installing PyTorch (CPU-only for HarmonyOS)..."
echo "Note: GPU support on HarmonyOS is limited"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install package in development mode
echo ""
echo "Installing NVFP4-DDIM Optimizer..."
pip install -e .

# Install development dependencies
echo ""
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install quality metrics dependencies
echo ""
read -p "Install quality metrics dependencies (FID, LPIPS, CLIP)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing quality metrics dependencies..."
    pip install lpips pytorch-msssim torchmetrics
    echo "Note: CLIP installation may take time on HarmonyOS"
    pip install git+https://github.com/openai/CLIP.git || echo "CLIP installation failed (optional)"
fi

# HarmonyOS-specific optimizations
echo ""
echo "Configuring HarmonyOS optimizations..."

# Set environment variables for optimal performance
cat >> venv/bin/activate << 'EOF'

# HarmonyOS optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export ATEN_CPU_CAPABILITY=default

# Model cache directory
export MODEL_CACHE_DIR="$HOME/Documents/diffusion-models"
mkdir -p "$MODEL_CACHE_DIR"

echo "HarmonyOS optimizations enabled"
EOF

echo "Environment variables configured"

# Verify installation
echo ""
echo "Verifying installation..."
python verify_setup.py || echo "Verification completed with warnings"

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "HarmonyOS-specific notes:"
echo "  - CPU-only mode enabled (recommended for HarmonyOS)"
echo "  - Memory optimizations configured"
echo "  - Use 'fast' preset for best performance"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "To run examples (CPU mode):"
echo "  python examples/basic_optimization.py --device cpu --preset fast"
echo ""
echo "For HarmonyOS-specific help, see:"
echo "  INSTALL_HARMONYOS.md"
echo ""
