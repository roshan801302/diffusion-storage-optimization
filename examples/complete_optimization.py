"""
Complete Optimization Example
Demonstrates all optimization techniques combined
"""

import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL
)
import time


def create_optimized_pipeline(
    model_id="stabilityai/stable-diffusion-2-1-base",
    use_fast_sampler=True,
    use_tuned_vae=True,
    enable_xformers=True,
    enable_compile=False,
    device="cuda"
):
    """
    Create a fully optimized Stable Diffusion pipeline
    
    Args:
        model_id: HuggingFace model identifier
        use_fast_sampler: Use DPM-Solver++ instead of DDIM
        use_tuned_vae: Use fine-tuned VAE for better quality
        enable_xformers: Enable memory-efficient attention
        enable_compile: Use torch.compile (PyTorch 2.0+)
        device: Device to run on
    
    Returns:
        Optimized pipeline ready for inference
    """
    
    print("Creating optimized pipeline...")
    
    # Choose scheduler
    if use_fast_sampler:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id,
            subfolder="scheduler"
        )
        print("✓ Using DPM-Solver++ (fast sampler)")
    else:
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler"
        )
        print("✓ Using DDIM")
    
    # Load VAE
    if use_tuned_vae:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
        )
        print("✓ Using fine-tuned VAE")
    else:
        vae = None
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        vae=vae,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    pipe = pipe.to(device)
    print(f"✓ Loaded model in FP16 on {device}")
    
    # Enable optimizations
    pipe.enable_attention_slicing()
    print("✓ Enabled attention slicing")
    
    pipe.enable_vae_slicing()
    print("✓ Enabled VAE slicing")
    
    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✓ Enabled xFormers")
        except Exception as e:
            print(f"⚠ xFormers not available: {e}")
    
    if enable_compile:
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
            print("✓ Compiled U-Net")
        except Exception as e:
            print(f"⚠ torch.compile not available: {e}")
    
    return pipe


def generate_with_metrics(pipe, prompt, num_steps=50, **kwargs):
    """Generate image and measure performance"""
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        **kwargs
    ).images[0]
    
    elapsed = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    metrics = {
        "time": elapsed,
        "peak_memory": peak_memory,
        "steps_per_sec": num_steps / elapsed
    }
    
    return image, metrics


def main():
    # Configuration
    prompts = [
        "a beautiful landscape with mountains and a lake at sunset",
        "a futuristic city with flying cars and neon lights",
        "a cute robot playing with a cat in a garden"
    ]
    
    # Create optimized pipeline
    pipe = create_optimized_pipeline(
        use_fast_sampler=True,
        use_tuned_vae=True,
        enable_xformers=True,
        enable_compile=False  # Set to True for PyTorch 2.0+
    )
    
    print("\n" + "="*60)
    print("Generating images...")
    print("="*60)
    
    # Generate images
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        image, metrics = generate_with_metrics(
            pipe,
            prompt=prompt,
            num_inference_steps=20,  # Fast generation
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=torch.Generator("cuda").manual_seed(42 + i)
        )
        
        # Save image
        filename = f"output_{i+1}.png"
        image.save(filename)
        
        # Print metrics
        print(f"  Time: {metrics['time']:.2f}s")
        print(f"  Peak Memory: {metrics['peak_memory']:.2f} GB")
        print(f"  Speed: {metrics['steps_per_sec']:.1f} steps/sec")
        print(f"  Saved: {filename}")
    
    print("\n" + "="*60)
    print("Complete! All optimizations applied successfully.")
    print("="*60)


if __name__ == "__main__":
    main()
