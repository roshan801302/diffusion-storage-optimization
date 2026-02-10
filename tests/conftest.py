"""Pytest configuration and shared fixtures."""

import pytest
import torch
import torch.nn as nn
from hypothesis import strategies as st
from typing import Tuple


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 8 * 8, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    return SimpleModel()


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(64, 64)


@pytest.fixture
def sample_image_batch():
    """Create a batch of sample images."""
    return torch.randn(4, 3, 32, 32)


# Hypothesis strategies for property-based testing

@st.composite
def tensor_strategy(draw, min_size=1, max_size=100):
    """Generate random tensors for property testing."""
    rows = draw(st.integers(min_value=min_size, max_value=max_size))
    cols = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate tensor with reasonable value range
    values = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=rows * cols,
            max_size=rows * cols
        )
    )
    
    tensor = torch.tensor(values).reshape(rows, cols)
    return tensor


@st.composite
def quantization_config_strategy(draw):
    """Generate random quantization configurations."""
    from src.nvfp4_ddim_optimizer.quantization import QuantizationConfig
    
    strategy = draw(st.sampled_from(["per_channel", "per_tensor"]))
    calibration_method = draw(st.sampled_from(["minmax", "percentile", "entropy"]))
    num_samples = draw(st.integers(min_value=10, max_value=500))
    percentile = draw(st.floats(min_value=90.0, max_value=99.99))
    
    return QuantizationConfig(
        enabled=True,
        strategy=strategy,
        calibration_method=calibration_method,
        num_calibration_samples=num_samples,
        percentile=percentile
    )


@st.composite
def sampling_config_strategy(draw):
    """Generate random sampling configurations."""
    from src.nvfp4_ddim_optimizer.sampling import SamplingConfig
    
    num_steps = draw(st.integers(min_value=10, max_value=200))
    schedule_type = draw(st.sampled_from(["uniform", "quadratic"]))
    eta = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return SamplingConfig(
        num_inference_steps=num_steps,
        schedule_type=schedule_type,
        eta=eta
    )
