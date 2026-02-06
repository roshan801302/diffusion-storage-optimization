"""
Interactive Demo Script
Can be converted to Jupyter notebook for interactive exploration
"""

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def setup_pipeline(optimization_level="medium"):
    """
    Setup pipeline with different optimization levels
    
    Args:
        optimization_level: "low", "medium", or "high"
    """
    model_id = "stabilityai/stable-diffusion-2-1-base"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    if optimization_level in ["medium", "high"]:
        pipe.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler"
        )
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    
    if optimization_level == "high":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xFormers not available")
    
    return pipe


def generate_comparison(pipe, prompt, step_counts=[20, 50, 100]):
    """Generate images with different step counts for comparison"""
    
    images = []
    times = []
    
    for steps in step_counts:
        import time
        start = time.time()
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        elapsed = time.time() - start
        images.append(image)
        times.append(elapsed)
    
    return images, times


def visualize_comparison(images, times, step_counts):
    """Visualize generated images side by side"""
    
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    
    for i, (img, t, steps) in enumerate(zip(images, times, step_counts)):
        axes[i].imshow(img)
        axes[i].set_title(f"{steps} steps\n{t:.2f}s")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def explore_guidance_scale(pipe, prompt, guidance_scales=[3.0, 7.5, 15.0]):
    """Explore effect of guidance scale"""
    
    images = []
    
    for scale in guidance_scales:
        image = pipe(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=scale,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        images.append(image)
    
    # Visualize
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    
    for i, (img, scale) in enumerate(zip(images, guidance_scales)):
        axes[i].imshow(img)
        axes[i].set_title(f"Guidance: {scale}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("guidance_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main interactive demo"""
    
    print("="*60)
    print("Interactive Diffusion Optimization Demo")
    print("="*60)
    
    # Setup
    print("\n1. Setting up optimized pipeline...")
    pipe = setup_pipeline(optimization_level="high")
    
    # Test prompt
    prompt = "a serene landscape with mountains, lake, and sunset, highly detailed"
    
    # Compare step counts
    print("\n2. Comparing different step counts...")
    images, times = generate_comparison(pipe, prompt, step_counts=[20, 50, 100])
    visualize_comparison(images, times, [20, 50, 100])
    print("✓ Saved comparison.png")
    
    # Explore guidance
    print("\n3. Exploring guidance scales...")
    explore_guidance_scale(pipe, prompt, guidance_scales=[3.0, 7.5, 15.0])
    print("✓ Saved guidance_comparison.png")
    
    print("\n" + "="*60)
    print("Demo complete! Check the generated images.")
    print("="*60)


if __name__ == "__main__":
    main()
