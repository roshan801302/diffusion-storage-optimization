# Latent Diffusion and Perceptual Compression

## The Latent Diffusion Breakthrough

**Key Insight**: Run diffusion in a compressed latent space instead of pixel space.

### Architecture
```
Image → VAE Encoder → Latent (compressed) → Diffusion Process → VAE Decoder → Image
         [E]            [z]                    [U-Net]            [D]
```

## Perceptual Compression with VAE

**VAE (Variational Autoencoder)** provides:
- Perceptually-aware compression (preserves visual quality)
- Downsampling factor **f** (commonly 8 or 16)
- Compact latent representation

### Memory and Compute Savings

| Downsampling f | Spatial Reduction | Memory Reduction | Compute Reduction |
|----------------|-------------------|------------------|-------------------|
| f=4 | 16× | ~16× | ~16× |
| f=8 | 64× | ~64× | ~50-60× |
| f=16 | 256× | ~256× | ~200× |

**Example**: 512×512 RGB image
- Pixel space: 512×512×3 = 786,432 values
- Latent space (f=8): 64×64×4 = 16,384 values
- **48× fewer values to process**

## Enables Megapixel Synthesis

With latent diffusion:
- 1024×1024 images on 8GB GPUs (impossible in pixel space)
- 2048×2048 images on 24GB GPUs
- Real-time generation becomes feasible

## VAE Architecture and Tradeoffs

### Low-f, Low-Channels (Speed Priority)
```python
# Fast but may lose fine details
f = 4
latent_channels = 4
encoder_channels = [64, 128, 256, 512]
```

### High-Channels (Quality Priority)
```python
# Slower but preserves fine details
f = 8
latent_channels = 4
encoder_channels = [128, 256, 512, 512]
```

## VAE Best Practices

### 1. Evaluate Reconstruction Quality

```python
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# FID: Distribution similarity
fid = FrechetInceptionDistance(feature=2048)
fid.update(real_images, fake=False)
fid.update(reconstructed_images, fake=True)
print(f"FID: {fid.compute()}")

# LPIPS: Perceptual similarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
score = lpips(original, reconstructed)
print(f"LPIPS: {score}")

# MS-SSIM: Multi-scale structural similarity
from pytorch_msssim import ms_ssim
score = ms_ssim(original, reconstructed, data_range=1.0)
print(f"MS-SSIM: {score}")
```

### 2. Tiled Decoding for Large Images

Process decoder in tiles to avoid memory spikes:

```python
def tiled_decode(vae, latents, tile_size=64, overlap=8):
    """Decode large latents in tiles to save memory"""
    B, C, H, W = latents.shape
    decoded_tiles = []
    
    for i in range(0, H, tile_size - overlap):
        for j in range(0, W, tile_size - overlap):
            tile = latents[:, :, i:i+tile_size, j:j+tile_size]
            decoded_tile = vae.decode(tile)
            decoded_tiles.append((i, j, decoded_tile))
    
    # Blend overlapping regions
    return blend_tiles(decoded_tiles, H*8, W*8, overlap*8)
```

### 3. VAE Fine-Tuning

For domain-specific data, fine-tune VAE:
```python
# Fine-tune on your dataset
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)

for images in dataloader:
    latents = vae.encode(images).latent_dist.sample()
    reconstructed = vae.decode(latents)
    
    # Perceptual loss + KL divergence
    loss = lpips_loss(images, reconstructed) + kl_weight * kl_divergence
    loss.backward()
    vae_optimizer.step()
```

## Latent Space Properties

### Advantages
- **Smooth interpolation**: Linear interpolation in latent space produces coherent images
- **Semantic editing**: Latent directions correspond to semantic attributes
- **Efficient conditioning**: Text/class embeddings easily integrated

### Considerations
- **High-frequency details**: Some fine details may be lost (mitigated with higher VAE capacity)
- **Decoder artifacts**: VAE decoder can introduce artifacts (improved with training)
- **Latent statistics**: Diffusion must match latent distribution (use scaling factors)

## Implementation: Stable Diffusion Architecture

```python
from diffusers import AutoencoderKL, UNet2DConditionModel

# VAE: 512×512 → 64×64×4 (f=8)
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse"  # Fine-tuned VAE
)

# U-Net operates on 64×64×4 latents
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="unet"
)

# Encode image to latent
with torch.no_grad():
    latent = vae.encode(image).latent_dist.sample()
    latent = latent * 0.18215  # Scaling factor

# Run diffusion in latent space (64× smaller!)
# ... diffusion process ...

# Decode back to pixel space
with torch.no_grad():
    decoded = vae.decode(latent / 0.18215).sample
```

## Memory Comparison

| Approach | Working Memory | Peak Memory | Relative |
|----------|----------------|-------------|----------|
| Pixel Diffusion (512²) | 12GB | 16GB | 1.0× |
| Latent Diffusion (f=8) | 2GB | 4GB | 0.25× |
| Latent Diffusion (f=16) | 0.8GB | 1.5GB | 0.09× |

## Advanced: Cascaded Latent Diffusion

For ultra-high resolution:
```
64×64 latent → 256×256 latent → 1024×1024 latent → 8192×8192 image
   (base)         (upsampler)        (upsampler)        (decoder)
```

Each stage operates at manageable resolution.

## Next Steps

Combine latent diffusion with:
- DDIM sampling (multiplicative speedup: 64× spatial × 20× temporal = 1280× total)
- Quantization (further memory reduction)
- Efficient attention mechanisms
