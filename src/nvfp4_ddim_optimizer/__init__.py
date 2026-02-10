"""
NVFP4-DDIM Optimizer

A comprehensive optimization suite for diffusion models combining NVIDIA's FP4 
quantization with DDIM sampling for maximum storage reduction and inference acceleration.
"""

__version__ = "0.1.0"

from .quantization import NVFP4Quantizer, QuantizedTensor, QuantizedModel
from .sampling import DDIMScheduler, TimestepScheduler
from .pipeline import OptimizationPipeline, QuantizationConfig, SamplingConfig
from .metrics import QualityMetrics, PerformanceMetrics

__all__ = [
    "NVFP4Quantizer",
    "QuantizedTensor",
    "QuantizedModel",
    "DDIMScheduler",
    "TimestepScheduler",
    "OptimizationPipeline",
    "QuantizationConfig",
    "SamplingConfig",
    "QualityMetrics",
    "PerformanceMetrics",
]
