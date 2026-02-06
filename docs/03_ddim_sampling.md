# Deterministic Accelerated Sampling (DDIM)

## Problem: Slow DDPM Sampling

Standard DDPM (Denoising Diffusion Probabilistic Models):
- Requires 1000+ timesteps for high-quality generation
- Each step = one full U-Net forward pass
- Total inference time: 30-60 seconds on modern GPUs

## DDIM Solution

**DDIM (Denoising Diffusion Implicit Models)** introduces:
- Non-Markovian deterministic sampling path
- Set stochasticity parameter η → 0 for deterministic generation
- Skip timesteps without quality loss

### Key Benefits
- **4×-20× fewer steps** (50-250 steps vs 1000)
- **~30% lower peak memory** (fewer intermediate activations)
- **Deterministic**: same seed → same output

### Tradeoffs
- Quality (FID) may drop at extreme step reduction (<20 steps)
- Some loss of diversity compared to stochastic sampling
- Requires careful noise schedule tuning

## Mathematical Formulation

DDPM update:
```
x_{t-1} = √(α_{t-1}) · pred_x0 + √(1-α_{t-1}-σ²) · pred_ε + σ · z
```

DDIM update (η=0):
```
x_{t-1} = √(α_{t-1}) · pred_x0 + √(1-α_{t-1}) · pred_ε
```

The deterministic path allows skipping timesteps: t → t-k

## Practical Sampling Strategies

### 1. Start with 20-100 Steps
```python
# Conservative: 50 steps (good quality/speed balance)
num_inference_steps = 50

# Aggressive: 20 steps (faster, slight quality loss)
num_inference_steps = 20

# High quality: 100 steps (minimal difference from 1000)
num_inference_steps = 100
```

### 2. Hybrid Samplers
Combine DDIM with other accelerators:
- **DPM-Solver**: 10-20 steps, better quality than DDIM
- **UniPC**: Unified predictor-corrector, 5-10 steps
- **LCM (Latent Consistency Models)**: 1-4 steps

### 3. Adaptive Step Allocation
Allocate more steps to critical denoising phases:
```python
# More steps early (structure formation)
# Fewer steps late (detail refinement)
timesteps = [999, 950, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 25, 10, 5, 1]
```

### 4. Early Exit Heuristics
Monitor generation quality and stop when sufficient:
```python
if perceptual_quality_metric(x_t) > threshold:
    break  # Stop early, save compute
```

## Implementation Example

```python
from diffusers import DDIMScheduler, StableDiffusionPipeline

# Load model with DDIM scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    scheduler=DDIMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="scheduler"
    )
)

# Generate with 50 steps (vs 1000 default)
image = pipe(
    prompt="a photo of an astronaut riding a horse",
    num_inference_steps=50,  # 20× speedup
    eta=0.0  # Deterministic (DDIM)
).images[0]
```

## Performance Comparison

| Sampler | Steps | Time (s) | FID ↓ | Memory (GB) |
|---------|-------|----------|-------|-------------|
| DDPM | 1000 | 45.2 | 12.3 | 8.2 |
| DDIM | 100 | 5.1 | 12.8 | 5.9 |
| DDIM | 50 | 2.8 | 13.5 | 5.7 |
| DDIM | 20 | 1.2 | 15.2 | 5.6 |
| DPM-Solver++ | 20 | 1.3 | 13.1 | 5.6 |

*Benchmark: Stable Diffusion 1.5, 512×512, A100 GPU*

## Best Practices

1. **Start with 50 steps** for production deployments
2. **Use DPM-Solver++** for <20 steps (better than DDIM)
3. **Profile your specific use case** - optimal steps vary by model
4. **Consider quality metrics** - FID, CLIP score, human evaluation
5. **Combine with other optimizations** - latent diffusion, FP16, etc.

## Advanced: Custom Schedulers

```python
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
    prediction_type="epsilon"
)

# Custom timestep schedule
scheduler.set_timesteps(num_inference_steps=50)
```

## Next Steps

Combine DDIM with:
- Latent diffusion (next section) for multiplicative speedup
- Quantization for memory reduction
- Compiled models for additional acceleration
