"""Setup script for NVFP4-DDIM Optimizer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="nvfp4-ddim-optimizer",
    version="0.1.0",
    author="rr",
    author_email="rr@example.com",
    description="Storage and memory optimization suite for diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roshan801302/diffusion-storage-optimization/tree/main",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Other OS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "hypothesis>=6.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "metrics": [
            "lpips>=0.1.4",
            "pytorch-msssim>=1.0.0",
            "torchmetrics>=1.0.0",
            "clip @ git+https://github.com/openai/CLIP.git",
        ],
    },
)
