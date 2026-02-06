"""
Latent Diffusion Analysis
Demonstrate memory and compute savings from latent space diffusion
"""

import torch
import time
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np


def analyze_vae_compression(vae, image_path, device="cuda"):
    """Analyze VAE compression characteristics"""
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    image_tensor = (image_tensor - 0.5) * 2  # Normalize to [-1, 1]
    
    print(f"Original image shape: {image_tensor.shape}")
    print(f"Original size: {image_tensor.numel() * 4 / 1e6:.2f} MB (FP32)")
    
    # Encode to latent
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * 0.18215  # Scaling factor
    
    print(f"\nLatent shape: {latent.shape}")
    print(f"Latent size: {latent.numel() * 4 / 1e6:.2f} MB (FP32)")
    
    # Calculate compression ratio
    compression_ratio = image_tensor.numel() / latent.numel()
    print(f"\nCompression ratio: {compression_ratio:.1f}×")
    
    # Decode back
    with torch.no_grad():
        reconstructed = vae.decode(latent / 0.18215).sample
    
    # Calculate reconstruction quality
    mse = torch.mean((image_tensor - reconstructed) ** 2).item()
    psnr = 10 * np.log10(4.0 / mse)  # Range is [-1, 1], so max diff is 2
    
    print(f"\nReconstruction PSNR: {psnr:.2f} dB")
    
    # Save reconstructed image
    reconstructed_np = ((reconstructed[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    reconstructed_img = Image.fromarray(reconstructed_np)
    
    return {
        "compression_ratio": compression_ratio,
        "psnr": psnr,
        "latent_shape": latent.shape,
        "reconstructed": reconstructed_img
    }


def compare_resolutions(vae, device="cuda"):
    """Compare memory requirements at different resolutions"""
    
    resolutions = [256, 512, 768, 1024]
    
    print("\n" + "="*60)
    print("Memory Requirements by Resolution")
    print("="*60)
    print(f"{'Resolution':<12} {'Pixel Space':<15} {'Latent Space':<15} {'Ratio':<10}")
    print("-"*60)
    
    for res in resolutions:
        # Pixel space
        pixel_size = res * res * 3 * 4 / 1e9  # RGB, FP32, in GB
        
        # Latent space (f=8 downsampling, 4 channels)
        latent_res = res // 8
        latent_size = latent_res * latent_res * 4 * 4 / 1e9  # 4 channels, FP32, in GB
        
        ratio = pixel_size / latent_size
        
        print(f"{res}×{res:<7} {pixel_size:.3f} GB{'':<7} {latent_size:.3f} GB{'':<7} {ratio:.1f}×")


def benchmark_diffusion_step(latent_shape, device="cuda"):
    """Estimate compute for one diffusion step"""
    
    # Simulate U-Net forward pass
    batch_size, channels, height, width = latent_shape
    
    # Rough FLOP estimate for U-Net
    # Simplified: conv layers dominate
    flops_per_conv = height * width * channels * channels * 9  # 3×3 conv
    num_conv_layers = 40  # Typical U-Net
    total_flops = flops_per_conv * num_conv_layers
    
    print(f"\nEstimated FLOPs per diffusion step: {total_flops / 1e9:.2f} GFLOPs")
    
    # Memory estimate
    activation_memory = batch_size * channels * height * width * 4 * 10  # 10 layers stored
    print(f"Estimated activation memory: {activation_memory / 1e9:.2f} GB")


def main():
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Analyze compression
    print("\n" + "="*60)
    print("VAE Compression Analysis")
    print("="*60)
    
    # Create a sample image if none exists
    sample_image = Image.new("RGB", (512, 512), color=(100, 150, 200))
    sample_image.save("sample_input.png")
    
    results = analyze_vae_compression(vae, "sample_input.png")
    results["reconstructed"].save("sample_reconstructed.png")
    
    # Compare resolutions
    compare_resolutions(vae)
    
    # Benchmark diffusion step
    print("\n" + "="*60)
    print("Diffusion Step Analysis (512×512 image)")
    print("="*60)
    benchmark_diffusion_step((1, 4, 64, 64))  # Latent shape for 512×512


if __name__ == "__main__":
    main()
