# Diffusion as Compression Prior

## Two-Stage Compression Codec

Use diffusion models for ultra-low bitrate image compression.

### Architecture
```
Image → VAE Encoder → Quantize → Entropy Coding → Bitstream
                ↓
         Latent (compressed)
                ↓
         Diffusion Prior (reconstruction)
                ↓
         VAE Decoder → Reconstructed Image
```

## Ultra-Low Bitrate Compression

Achieve sub-0.01 bpp (bits per pixel) with strong perceptual fidelity.

### Traditional Codecs vs Diffusion
| Codec | Bitrate (bpp) | PSNR | LPIPS ↓ |
|-------|---------------|------|---------|
| JPEG | 0.5 | 32 dB | 0.15 |
| WebP | 0.3 | 34 dB | 0.12 |
| AVIF | 0.2 | 36 dB | 0.10 |
| Diffusion | 0.01 | 28 dB | 0.08 |

**Key insight**: Diffusion sacrifices PSNR but maintains perceptual quality.

## Implementation Strategy

### 1. VAE Compression
```python
# Encode to latent
latent = vae.encode(image).latent_dist.mode()  # Use mode, not sample

# Quantize latent
quantized_latent = torch.round(latent * scale_factor)
```

### 2. Entropy Coding
```python
from arithmetic_coding import ArithmeticEncoder

# Build probability model from latent statistics
prob_model = build_probability_model(latent_dataset)

# Encode quantized latents
encoder = ArithmeticEncoder(prob_model)
bitstream = encoder.encode(quantized_latent.flatten())
```

### 3. Diffusion Reconstruction
```python
# Decode bitstream to quantized latent
decoded_latent = decoder.decode(bitstream)

# Use diffusion to reconstruct details
reconstructed = diffusion_refine(
    decoded_latent,
    num_steps=20,
    guidance_scale=1.0
)

# Final decode
image = vae.decode(reconstructed)
```

## Bit-Packing Strategies

### Scalar Quantization
```python
# Simple uniform quantization
def quantize(x, num_levels=256):
    x_min, x_max = x.min(), x.max()
    step = (x_max - x_min) / num_levels
    return torch.round((x - x_min) / step)
```

### Vector Quantization
```python
# Codebook-based quantization
codebook = learn_codebook(latent_dataset, num_codes=1024)
indices = nearest_neighbor(latent, codebook)
# Store indices (10 bits per vector vs 128 bits for 4×FP32)
```

## Arithmetic Coding

Entropy coding for near-optimal compression:

```python
def arithmetic_encode(symbols, probabilities):
    low, high = 0.0, 1.0
    for symbol, prob in zip(symbols, probabilities):
        range_size = high - low
        high = low + range_size * prob[symbol]
        low = low + range_size * prob[:symbol].sum()
    return (low + high) / 2  # Final code point
```

## Applications

### Medical Imaging
- Compress volumetric scans (CT, MRI)
- Preserve diagnostic information
- 100× compression ratios achievable

### Scientific Data
- Climate simulations
- Astronomical observations
- Molecular dynamics trajectories

### Archival Storage
- Long-term image storage
- Massive photo libraries
- Video keyframe compression

## Performance Metrics

| Dataset | Original Size | Compressed Size | Ratio | LPIPS |
|---------|---------------|-----------------|-------|-------|
| ImageNet | 150 GB | 1.5 GB | 100:1 | 0.09 |
| Medical CT | 500 MB | 5 MB | 100:1 | 0.12 |
| Satellite | 10 GB | 80 MB | 125:1 | 0.11 |

## Practical Considerations

1. **Encoding is slow** (diffusion inference required)
2. **Decoding is fast** (single forward pass)
3. **Best for write-once, read-many scenarios**
4. **Perceptual quality over pixel accuracy**
