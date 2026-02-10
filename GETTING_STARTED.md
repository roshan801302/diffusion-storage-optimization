# Getting Started

## Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/roshan801302/diffusion-storage-optimization.git
cd diffusion-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Your First Benchmark

```bash
# Test memory optimizations
python benchmarks/memory_benchmark.py
```

This will compare different optimization strategies and show you the memory and speed improvements.

### 3. Generate Optimized Images

```bash
# Run complete optimization example
python examples/complete_optimization.py
```

This demonstrates all optimizations combined and generates sample images.

### 4. Explore the Documentation

Start with these docs in order:
1. `docs/01_introduction.md` - Overview and motivation
2. `docs/03_ddim_sampling.md` - Fast sampling strategies
3. `docs/04_latent_diffusion.md` - Memory-efficient generation
4. `docs/10_recommendations.md` - Practical implementation guide

## Common Use Cases

### Use Case 1: I want to generate images faster

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="scheduler"
    )
).to("cuda")

# Generate in ~1 second
image = pipe("a cat", num_inference_steps=20).images[0]
```

### Use Case 2: I'm running out of GPU memory

```python
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

# Now works on 4-6GB GPUs
image = pipe("a landscape", height=1024, width=1024).images[0]
```

### Use Case 3: I want maximum quality

```python
from diffusers import DDIMScheduler

pipe.scheduler = DDIMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="scheduler",
    beta_schedule="cosine"
)

# High quality generation
image = pipe(
    "a beautiful landscape",
    num_inference_steps=100,
    guidance_scale=12.0
).images[0]
```

## Next Steps

- Read the full documentation in `docs/`
- Check out `presentation_outline.md` for a structured overview
- Experiment with different optimization combinations
- Run benchmarks on your specific hardware
- Adapt the code for your use case

## Troubleshooting

### Out of Memory Error
- Enable CPU offloading: `pipe.enable_model_cpu_offload()`
- Reduce resolution or batch size
- Use attention slicing: `pipe.enable_attention_slicing()`

### Slow Generation
- Reduce inference steps (try 20-50)
- Use DPM-Solver++ scheduler
- Enable xFormers: `pipe.enable_xformers_memory_efficient_attention()`

### Poor Quality
- Increase inference steps (50-100)
- Adjust guidance scale (7.5-12.0)
- Use cosine noise schedule
- Try fine-tuned VAE

## Resources

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [DDIM Paper](https://arxiv.org/abs/2010.02502)
- [Latent Diffusion Paper](https://arxiv.org/abs/2112.10752)
