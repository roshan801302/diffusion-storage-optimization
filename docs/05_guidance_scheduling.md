# Guidance, Scheduling, and Architecture Variants

## Classifier-Free Guidance (CFG)

### Concept
Steer generation toward conditioning (e.g., text prompt) without a separate classifier.

### Formula
```
ε_guided = ε_uncond + w · (ε_cond - ε_uncond)
```

Where:
- `ε_uncond`: Unconditional noise prediction
- `ε_cond`: Conditional noise prediction (with prompt)
- `w`: Guidance scale (typically 7-15)

### Tradeoffs

**Benefits:**
- Stronger prompt adherence
- Higher quality generations
- No separate classifier needed

**Costs:**
- **2× compute per step** (two U-Net passes)
- Slightly higher memory usage
- Can reduce diversity at high guidance scales

### Optimization Strategies

```python
# Standard CFG (2 forward passes)
noise_pred_uncond = unet(latent, t, encoder_hidden_states=null_embed)
noise_pred_cond = unet(latent, t, encoder_hidden_states=text_embed)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

# Optimized: Batch both passes
latent_model_input = torch.cat([latent] * 2)
text_embeddings = torch.cat([null_embed, text_embed])
noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
```

### Adaptive Guidance
Reduce guidance scale over time to save compute:
```python
# High guidance early (structure), low guidance late (details)
guidance_scale = 15.0 * (t / T) + 5.0
```

## Noise Schedules

### Linear Schedule (Original DDPM)
```python
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps)
```

**Issues:**
- Too much noise early (destroys global structure)
- Inefficient use of timesteps

### Cosine Schedule (Improved)
```python
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

**Benefits:**
- Better preservation of global structure
- More efficient denoising trajectory
- Often better FID scores

### Comparison

| Schedule | FID ↓ | Global Structure | Detail Quality |
|----------|-------|------------------|----------------|
| Linear | 15.2 | Moderate | Good |
| Cosine | 13.8 | Excellent | Good |
| Scaled Linear | 14.5 | Good | Excellent |

## Architecture Variants

### 1. U-Net-NAS (Neural Architecture Search)

Optimize U-Net architecture for efficiency:
- Reduce channels in less critical layers
- Adjust attention resolution thresholds
- Optimize skip connection patterns

**Results:**
- 30-40% fewer parameters
- 20-30% faster inference
- Minimal quality loss (<1% FID increase)

### 2. DiT (Diffusion Transformer)

Replace U-Net with transformer architecture:

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size)
        
    def forward(self, x, c):
        # Adaptive layer norm with conditioning
        x = x + self.attn(self.norm1(x, c))
        x = x + self.mlp(self.norm2(x, c))
        return x
```

**Advantages:**
- Better scalability (scales well with compute)
- Unified architecture (no encoder/decoder split)
- Strong performance at large scale

**Tradeoffs:**
- Requires more compute for small models
- Less inductive bias than U-Net
- Quadratic attention complexity

### 3. Efficient Attention Mechanisms

#### Flash Attention
```python
from flash_attn import flash_attn_func

# Standard attention: O(n²) memory
attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

# Flash attention: O(n) memory, same result
attn_output = flash_attn_func(q, k, v)
```

**Benefits:**
- 2-4× faster attention
- 10-20× lower memory
- Exact same output

#### xFormers Memory-Efficient Attention
```python
from xformers.ops import memory_efficient_attention

attn_output = memory_efficient_attention(q, k, v, attn_bias=None)
```

**Benefits:**
- Automatic kernel selection
- Works with various sequence lengths
- Easy drop-in replacement

## Practical Recommendations

### For Speed Priority
```python
scheduler = DDIMScheduler(beta_schedule="scaled_linear")
num_inference_steps = 20
guidance_scale = 7.5
use_flash_attention = True
```

### For Quality Priority
```python
scheduler = DDIMScheduler(beta_schedule="cosine")
num_inference_steps = 50
guidance_scale = 12.0
use_flash_attention = True
```

### For Memory Efficiency
```python
scheduler = DDIMScheduler(beta_schedule="cosine")
num_inference_steps = 50
guidance_scale = 7.5  # Lower = less compute
enable_attention_slicing = True
enable_vae_slicing = True
```

## Combined Optimization Example

```python
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    scheduler=DDIMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="scheduler",
        beta_schedule="cosine"
    )
)

# Enable all optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
pipe = pipe.to("cuda")

# Generate with optimized settings
image = pipe(
    prompt="a beautiful landscape",
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]
```

## Performance Impact

| Optimization | Speed | Memory | Quality |
|--------------|-------|--------|---------|
| Cosine schedule | +5% | 0% | +8% FID |
| Flash Attention | +40% | -60% | 0% |
| Lower guidance (7.5→5.0) | +50% | -10% | -5% adherence |
| DiT (large scale) | -20% | +30% | +10% FID |

*Relative to baseline Stable Diffusion 1.5*
