# U-Net Anatomy and Memory Hotspots

## U-Net Architecture Overview

The U-Net is the backbone of most diffusion models, consisting of:

```
Input (noisy latent/image)
    ↓
Encoder (downsampling blocks)
    ↓
Bottleneck (highest channel count, lowest spatial resolution)
    ↓
Decoder (upsampling blocks) ← skip connections from encoder
    ↓
Output (predicted noise/denoised result)
```

## Memory Hotspots

### 1. Bottleneck Layer
- **Highest channel count** (often 1024-2048 channels)
- **Peak activation memory** occurs here
- Spatial resolution is lowest, but channel depth is maximum

### 2. Skip Connections
- Store encoder activations for decoder use
- Must keep multiple resolution levels in memory simultaneously
- Contributes significantly to peak memory usage

### 3. Attention Mechanisms
- Self-attention: O(n²) memory complexity where n = spatial dimensions
- Cross-attention: Additional memory for conditioning (text embeddings)
- Multiple attention heads multiply memory requirements

## Activation vs Parameter Memory

**Critical Insight**: Activations > Parameters for peak memory usage

| Component | Parameters | Activations (per sample) |
|-----------|------------|--------------------------|
| Conv layers | ~1-2GB | ~4-8GB (depends on resolution) |
| Attention | ~0.5GB | ~2-4GB (quadratic in resolution) |
| Skip connections | 0GB | ~2-3GB (stored encoder features) |

**Example**: Stable Diffusion 1.5
- Model parameters: ~860M (~1.7GB in FP16)
- Peak activation memory: ~6-10GB (512×512 generation)

## Memory Breakdown by Resolution

| Resolution | Latent Size | Activation Memory | Total VRAM |
|------------|-------------|-------------------|------------|
| 512×512 | 64×64×4 | ~6GB | ~8GB |
| 768×768 | 96×96×4 | ~13GB | ~16GB |
| 1024×1024 | 128×128×4 | ~24GB | ~28GB |

*Assumes latent diffusion with f=8 downsampling, FP16 precision*

## Optimization Targets

Based on this anatomy, key optimization opportunities:

1. **Reduce timesteps** → fewer U-Net passes (DDIM, DPM-Solver)
2. **Work in latent space** → smaller spatial dimensions (Latent Diffusion)
3. **Efficient attention** → reduce quadratic complexity (Flash Attention)
4. **Gradient checkpointing** → trade compute for memory (training)
5. **Precision reduction** → FP16/BF16/INT8 (inference)
6. **Tiled processing** → process image in chunks (very high resolution)

## Profiling Your Model

```python
import torch
from torch.profiler import profile, ProfilerActivity

# Profile memory usage
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True) as prof:
    output = model(input_tensor)

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

This reveals exactly where memory is allocated during inference.
