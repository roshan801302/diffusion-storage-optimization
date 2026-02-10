"""Data models for quantization."""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


@dataclass
class QuantizationConfig:
    """Configuration for NVFP4 quantization."""
    
    enabled: bool = True
    strategy: str = "per_channel"  # "per_channel" or "per_tensor"
    calibration_method: str = "minmax"  # "minmax", "percentile", or "entropy"
    num_calibration_samples: int = 100
    percentile: float = 99.99  # For percentile calibration method
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_strategies = ["per_channel", "per_tensor"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid quantization strategy '{self.strategy}'. "
                f"Valid options are: {valid_strategies}"
            )
        
        valid_methods = ["minmax", "percentile", "entropy"]
        if self.calibration_method not in valid_methods:
            raise ValueError(
                f"Invalid calibration method '{self.calibration_method}'. "
                f"Valid options are: {valid_methods}"
            )
        
        if self.num_calibration_samples < 1:
            raise ValueError(
                f"num_calibration_samples must be positive, got {self.num_calibration_samples}"
            )
        
        if not 0 < self.percentile <= 100:
            raise ValueError(
                f"percentile must be in range (0, 100], got {self.percentile}"
            )


class QuantizedTensor:
    """Container for quantized tensor data."""
    
    def __init__(
        self,
        data: torch.Tensor,  # 4-bit packed data
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        shape: Tuple[int, ...],
        strategy: str,
        dtype: str = "nvfp4"
    ):
        """
        Initialize quantized tensor.
        
        Args:
            data: Packed 4-bit quantized data
            scale: Scale factors for dequantization
            zero_point: Zero points for dequantization
            shape: Original tensor shape
            strategy: Quantization strategy used
            dtype: Data type identifier
        """
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
        self.shape = shape
        self.strategy = strategy
        self.dtype = dtype
    
    def dequantize(self) -> torch.Tensor:
        """
        Convert quantized data back to FP16/FP32.
        
        Returns:
            Dequantized tensor
        """
        # Unpack 4-bit data to int8
        unpacked = self._unpack_4bit(self.data)
        
        # Dequantize: x = scale * (x_q - zero_point)
        if self.strategy == "per_channel":
            # Reshape scale and zero_point for broadcasting
            scale = self.scale.view(-1, 1)
            zero_point = self.zero_point.view(-1, 1)
            dequantized = scale * (unpacked.float() - zero_point)
        else:  # per_tensor
            dequantized = self.scale * (unpacked.float() - self.zero_point)
        
        return dequantized.reshape(self.shape)
    
    def _unpack_4bit(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack 4-bit values from packed int8 tensor."""
        # Each int8 contains two 4-bit values
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        
        # Interleave high and low nibbles
        unpacked = torch.stack([high, low], dim=-1).flatten()
        
        # Trim to original size if needed
        return unpacked[:self.shape[0] * self.shape[1]]
    
    def size_bytes(self) -> int:
        """
        Calculate storage size in bytes.
        
        Returns:
            Total size including data and metadata
        """
        data_size = self.data.numel()  # 4 bits per value, packed in int8
        scale_size = self.scale.numel() * 2  # FP16
        zero_point_size = self.zero_point.numel() * 2  # FP16
        
        return data_size + scale_size + zero_point_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data.cpu().numpy().tolist(),
            "scale": self.scale.cpu().numpy().tolist(),
            "zero_point": self.zero_point.cpu().numpy().tolist(),
            "shape": list(self.shape),
            "strategy": self.strategy,
            "dtype": self.dtype,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizedTensor":
        """Create from dictionary."""
        return cls(
            data=torch.tensor(data["data"], dtype=torch.int8),
            scale=torch.tensor(data["scale"], dtype=torch.float16),
            zero_point=torch.tensor(data["zero_point"], dtype=torch.float16),
            shape=tuple(data["shape"]),
            strategy=data["strategy"],
            dtype=data.get("dtype", "nvfp4"),
        )


class QuantizedModel:
    """Container for quantized model."""
    
    def __init__(
        self,
        quantized_state_dict: Dict[str, QuantizedTensor],
        config: QuantizationConfig,
        metadata: Dict[str, Any]
    ):
        """
        Initialize quantized model.
        
        Args:
            quantized_state_dict: Dictionary of quantized tensors
            config: Quantization configuration used
            metadata: Additional metadata (sizes, timestamps, etc.)
        """
        self.quantized_state_dict = quantized_state_dict
        self.config = config
        self.metadata = metadata
    
    def save(self, path: str):
        """
        Save quantized model to disk.
        
        Args:
            path: Output file path
        """
        import json
        import pickle
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        save_data = {
            "version": "1.0.0",
            "config": {
                "enabled": self.config.enabled,
                "strategy": self.config.strategy,
                "calibration_method": self.config.calibration_method,
                "num_calibration_samples": self.config.num_calibration_samples,
                "percentile": self.config.percentile,
            },
            "metadata": self.metadata,
            "state_dict": {
                name: tensor.to_dict()
                for name, tensor in self.quantized_state_dict.items()
            },
        }
        
        # Save as pickle for efficiency
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        
        print(f"Saved quantized model to {path}")
    
    @classmethod
    def load(cls, path: str) -> "QuantizedModel":
        """
        Load quantized model from disk.
        
        Args:
            path: Input file path
            
        Returns:
            Loaded quantized model
        """
        import pickle
        from pathlib import Path
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        # Verify version compatibility
        version = save_data.get("version", "0.0.0")
        if not version.startswith("1."):
            raise ValueError(
                f"Incompatible model version {version}. "
                f"This loader supports version 1.x.x"
            )
        
        # Reconstruct config
        config_data = save_data["config"]
        config = QuantizationConfig(**config_data)
        
        # Reconstruct state dict
        state_dict = {
            name: QuantizedTensor.from_dict(tensor_data)
            for name, tensor_data in save_data["state_dict"].items()
        }
        
        return cls(
            quantized_state_dict=state_dict,
            config=config,
            metadata=save_data["metadata"]
        )
    
    def to_pytorch(self) -> nn.Module:
        """
        Convert to PyTorch model with dequantization.
        
        Returns:
            PyTorch model ready for inference
        """
        # This will be implemented when we have the full model structure
        raise NotImplementedError(
            "to_pytorch() will be implemented in later tasks"
        )
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio vs FP32."""
        original_size = self.metadata.get("original_size_mb", 0)
        quantized_size = self.metadata.get("quantized_size_mb", 0)
        
        if quantized_size == 0:
            return 0.0
        
        return original_size / quantized_size
