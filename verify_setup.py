#!/usr/bin/env python3
"""Verify NVFP4-DDIM Optimizer setup on Linux."""

import sys
import platform


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("  ❌ Python 3.8+ required")
        return False
    print("  ✅ Python version OK")
    return True


def check_platform():
    """Check if running on supported platform."""
    print("\nChecking platform...")
    system = platform.system()
    print(f"  Platform: {system}")
    
    # Check for OpenKylin
    if system == "Linux":
        try:
            with open("/etc/os-release", "r") as f:
                os_info = f.read()
                if "OpenKylin" in os_info or "openKylin" in os_info:
                    print("  ✅ Running on OpenKylin")
                    print("  ℹ️  Full support with CUDA (x86_64) or CPU (ARM64)")
                    return True
        except FileNotFoundError:
            pass
        print("  ✅ Running on Linux")
    elif system == "Windows":
        print("  ✅ Running on Windows")
        print("  ℹ️  For best performance, consider using WSL2 or native CUDA")
    elif system == "Darwin":
        print("  ✅ Running on macOS")
        print("  ℹ️  CPU or Metal GPU support")
    else:
        print(f"  ⚠️  Running on {system} (experimental support)")
    
    return True


def check_pytorch():
    """Check PyTorch installation."""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print("  ✅ PyTorch OK")
        return True
    except ImportError:
        print("  ❌ PyTorch not installed")
        print("     Run: pip install torch torchvision")
        return False


def check_package():
    """Check package installation."""
    print("\nChecking nvfp4_ddim_optimizer package...")
    try:
        import nvfp4_ddim_optimizer
        print(f"  Package version: {nvfp4_ddim_optimizer.__version__}")
        
        # Check imports
        from nvfp4_ddim_optimizer import (
            QuantizationConfig,
            SamplingConfig,
            QuantizedTensor,
            QuantizedModel
        )
        print("  ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Package not installed: {e}")
        print("     Run: pip install -e .")
        return False


def check_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    try:
        from nvfp4_ddim_optimizer import QuantizationConfig, SamplingConfig
        
        # Test valid config
        config = QuantizationConfig(
            strategy="per_channel",
            calibration_method="minmax"
        )
        print("  ✅ Valid config creation OK")
        
        # Test invalid config
        try:
            invalid_config = QuantizationConfig(strategy="invalid")
            print("  ❌ Invalid config should raise error")
            return False
        except ValueError as e:
            print(f"  ✅ Invalid config rejected: {str(e)[:50]}...")
        
        return True
    except Exception as e:
        print(f"  ❌ Config validation failed: {e}")
        return False


def check_dependencies():
    """Check optional dependencies."""
    print("\nChecking optional dependencies...")
    
    deps = {
        "diffusers": "HuggingFace Diffusers",
        "transformers": "HuggingFace Transformers",
        "PIL": "Pillow (Image processing)",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "tqdm": "Progress bars",
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} not installed (optional)")
            all_ok = False
    
    return all_ok


def check_dev_tools():
    """Check development tools."""
    print("\nChecking development tools...")
    
    tools = {
        "pytest": "Testing framework",
        "hypothesis": "Property-based testing",
        "black": "Code formatter",
        "flake8": "Linter",
    }
    
    for module, name in tools.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} not installed (dev dependency)")
    
    return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("NVFP4-DDIM Optimizer Setup Verification")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_platform(),
        check_pytorch(),
        check_package(),
        check_config_validation(),
        check_dependencies(),
        check_dev_tools(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks[:5]):  # Core checks
        print("✅ Setup verification PASSED")
        print("\nYou're ready to start development!")
        print("\nNext steps:")
        print("  1. Read: SETUP_COMPLETE.md")
        print("  2. Run tests: make test")
        print("  3. Start Task 2: Implement NVFP4 quantization core")
        return 0
    else:
        print("❌ Setup verification FAILED")
        print("\nPlease fix the issues above and run again.")
        print("\nFor help, see: INSTALL_LINUX.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
