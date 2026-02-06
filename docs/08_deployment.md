# Framework and Deployment Optimizations

## Production Optimization Checklist

### 1. CPU Offloading
Move inactive model components to CPU to save VRAM.

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)

# Enable sequential CPU offloading
pipe.enable_sequential_cpu_offload()

# Or model-level offloading (more control)
pipe.enable_model_cpu_offload()
```

**Benefits:**
- Run on GPUs with limited VRAM (4-6GB)
- Slight latency increase (~10-20%)
- Enables larger batch sizes

### 2. torch.compile
JIT compilation for faster inference (PyTorch 2.0+).

```python
import torch

# Compile U-Net for faster inference
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

# First run is slow (compilation), subsequent runs are faster
image = pipe(prompt="a cat").images[0]  # ~30% faster after warmup
```

**Modes:**
- `default`: Balanced optimization
- `reduce-overhead`: Maximum speed, longer compile time
- `max-autotune`: Exhaustive search for best kernels

### 3. Tiled VAE Decoding
Process large images in tiles to avoid memory spikes.

```python
# Enable tiled VAE
pipe.enable_vae_tiling()

# Now can generate much larger images
image = pipe(
    prompt="landscape",
    height=2048,
    width=2048
).images[0]
```

### 4. Attention Slicing
Reduce memory for attention computation.

```python
# Slice attention computation
pipe.enable_attention_slicing(slice_size="auto")

# Or manual control
pipe.enable_attention_slicing(slice_size=1)  # More slices = less memory
```

### 5. xFormers Memory-Efficient Attention
Fastest attention implementation.

```python
# Requires: pip install xformers
pipe.enable_xformers_memory_efficient_attention()
```

**Benefits:**
- 2-4× faster attention
- 40-60% memory reduction
- No quality loss

### 6. BFloat16 Precision
Better numerical stability than FP16.

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.bfloat16
)
```

## Combined Optimization Example

```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Load with all optimizations
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    scheduler=DDIMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="scheduler"
    )
)

# Enable all optimizations
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

# Generate efficiently
image = pipe(
    prompt="a beautiful landscape",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=1024,
    width=1024
).images[0]
```

## Batch Processing

Process multiple prompts efficiently:

```python
prompts = ["a cat", "a dog", "a bird", "a fish"]

# Batch inference (faster than sequential)
images = pipe(
    prompt=prompts,
    num_inference_steps=50,
    guidance_scale=7.5
).images
```

## Avoiding VRAM Spikes

Combine offloading + tiled decode:

```python
# This combination prevents OOM on 8GB GPUs
pipe.enable_sequential_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()

# Can now generate 1024×1024 on 8GB VRAM
image = pipe(prompt="...", height=1024, width=1024).images[0]
```

## Performance Comparison

| Optimization | Speed | Memory | Complexity |
|--------------|-------|--------|------------|
| Baseline | 1.0× | 8GB | Low |
| + FP16 | 1.8× | 4GB | Low |
| + xFormers | 2.5× | 2.5GB | Medium |
| + torch.compile | 3.2× | 2.5GB | Medium |
| + CPU offload | 2.8× | 1.5GB | Low |
| + All combined | 3.5× | 1.5GB | High |

*Benchmark: SD 2.1, 512×512, 50 steps, A100*

## Production Deployment Patterns

### Pattern 1: High Throughput
```python
# Maximize batch size, use compilation
pipe.unet = torch.compile(pipe.unet)
pipe.enable_xformers_memory_efficient_attention()
batch_size = 8
```

### Pattern 2: Low Latency
```python
# Minimize steps, use fast samplers
scheduler = DPMSolverMultistepScheduler(...)
num_inference_steps = 20
```

### Pattern 3: Memory Constrained
```python
# Aggressive offloading and slicing
pipe.enable_sequential_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing(slice_size=1)
```

## Monitoring and Profiling

```python
import torch.cuda as cuda

# Monitor memory usage
print(f"Allocated: {cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {cuda.memory_reserved() / 1e9:.2f} GB")

# Profile inference
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    image = pipe(prompt="test").images[0]

print(prof.key_averages().table(sort_by="cuda_time_total"))
```
