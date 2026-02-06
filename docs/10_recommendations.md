# Tiered Recommendations

## Quick Reference Guide

Choose optimizations based on your constraints and priorities.

## Tier 1: Essential (Always Apply)

### Sampling: DDIM / Accelerated Samplers
```python
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

# DDIM: 50 steps (good balance)
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

# DPM-Solver++: 20 steps (faster, similar quality)
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
```

**Impact:** 10-20× speedup, 30% memory reduction

### Representation: Latent Diffusion with Tuned VAE
```python
# Use latent diffusion (f=8 or f=16)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
```

**Impact:** 50-200× compute reduction, enables high-resolution

### Numerical: FP16
```python
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
```

**Impact:** 50% memory, 1.5-2× speed, negligible quality loss

## Tier 2: Production (Recommended)

### Infrastructure: Offloading + Tiled Decoding
```python
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
```

**Impact:** Run on 4-6GB GPUs, handle larger images

### Attention: xFormers
```python
pipe.enable_xformers_memory_efficient_attention()
```

**Impact:** 2-4× faster attention, 40-60% memory reduction

### Compilation: torch.compile
```python
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
```

**Impact:** 20-30% additional speedup after warmup

## Tier 3: Advanced (Specialized Use Cases)

### INT8 Quantization with QAT
```python
# Requires quantization-aware training
quantized_model = load_quantized_model("model-int8-qat")
```

**Impact:** 75% memory, 2-3× speed, moderate quality loss

### Custom Schedulers
```python
scheduler = DDIMScheduler(
    beta_schedule="cosine",
    prediction_type="v_prediction"
)
```

**Impact:** Better quality at same step count

### Distilled Models
```python
# Use distilled student model
pipe = StableDiffusionPipeline.from_pretrained("distilled-sd-v1")
```

**Impact:** 2-4× faster, smaller model, slight quality loss

## Decision Matrix

### Use Case: Real-Time Generation (Latency Priority)
```python
# Configuration
scheduler = DPMSolverMultistepScheduler(...)
num_steps = 10-20
precision = torch.float16
enable_xformers = True
compile_model = True
```

**Expected:** <1 second per image (512×512)

### Use Case: High Quality (Quality Priority)
```python
# Configuration
scheduler = DDIMScheduler(beta_schedule="cosine")
num_steps = 100
precision = torch.float16
guidance_scale = 12.0
use_tuned_vae = True
```

**Expected:** Best quality, ~5 seconds per image

### Use Case: Memory Constrained (4-6GB VRAM)
```python
# Configuration
enable_cpu_offload = True
enable_vae_tiling = True
enable_attention_slicing = True
precision = torch.float16
num_steps = 50
```

**Expected:** Runs on low-end GPUs, moderate speed

### Use Case: Batch Processing (Throughput Priority)
```python
# Configuration
batch_size = 16
precision = torch.float16
enable_xformers = True
compile_model = True
num_steps = 50
```

**Expected:** Maximum images/second

## Complete Optimization Stack

```python
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
    DPMSolverMultistepScheduler
)

# Load with all Tier 1 + Tier 2 optimizations
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,  # Tier 1: FP16
    scheduler=DPMSolverMultistepScheduler.from_pretrained(  # Tier 1: Fast sampler
        "stabilityai/stable-diffusion-2-1",
        subfolder="scheduler"
    ),
    vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")  # Tier 1: Tuned VAE
)

# Tier 2: Infrastructure optimizations
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_xformers_memory_efficient_attention()
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

# Generate with optimized settings
image = pipe(
    prompt="a beautiful landscape",
    num_inference_steps=20,  # Tier 1: Reduced steps
    guidance_scale=7.5,
    height=1024,
    width=1024
).images[0]
```

## Performance Summary

| Configuration | Speed | Memory | Quality | Use Case |
|---------------|-------|--------|---------|----------|
| Baseline | 1× | 8GB | 100% | Reference |
| Tier 1 | 15× | 2GB | 98% | Standard |
| Tier 1+2 | 25× | 1.5GB | 98% | Production |
| Tier 1+2+3 | 40× | 1GB | 95% | Edge/Mobile |

## Monitoring and Validation

```python
# Measure performance
import time

start = time.time()
image = pipe(prompt="test").images[0]
elapsed = time.time() - start

print(f"Generation time: {elapsed:.2f}s")
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Validate quality
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=2048)
# Compare against baseline
```

## Next Steps

1. Start with Tier 1 optimizations (easy wins)
2. Profile your specific workload
3. Add Tier 2 based on bottlenecks
4. Consider Tier 3 for specialized needs
5. Continuously monitor and iterate
