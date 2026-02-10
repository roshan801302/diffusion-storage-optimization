#!/bin/bash
# Installation script for NVFP4-DDIM Optimizer on Linux

set -e  # Exit on error

echo "========================================="
echo "NVFP4-DDIM Optimizer Installation"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python >= 3.8
required_version="3.8"
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

# Install PyTorch (Linux-specific with CUDA support if available)
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

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
    pip install git+https://github.com/openai/CLIP.git
fi

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "To run examples:"
echo "  python examples/basic_optimization.py"
echo ""
