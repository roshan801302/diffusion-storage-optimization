"""
DDIM Sampling Demonstration
Compare DDPM vs DDIM sampling speeds and quality
"""

import torch
import time
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler
from PIL import Image


def benchmark_sampler(pipe, prompt, num_steps, sampler_name):
    """Benchmark a specific sampler configuration"""
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
    
    elapsed = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"{sampler_name} ({num_steps} steps):")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Peak Memory: {peak_memory:.2f} GB")
    print(f"  Speed: {num_steps/elapsed:.1f} steps/sec")
    
    return image, elapsed, peak_memory


def main():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    prompt = "a professional photograph of an astronaut riding a horse on mars"
    
    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda")
    
    results = []
    
    # Test 1: DDIM with 50 steps (recommended)
    print("\n" + "="*50)
    print("Test 1: DDIM 50 steps")
    print("="*50)
    pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    img1, t1, m1 = benchmark_sampler(pipe, prompt, 50, "DDIM")
    results.append(("ddim_50.png", img1))
    
    # Test 2: DDIM with 20 steps (aggressive)
    print("\n" + "="*50)
    print("Test 2: DDIM 20 steps")
    print("="*50)
    img2, t2, m2 = benchmark_sampler(pipe, prompt, 20, "DDIM")
    results.append(("ddim_20.png", img2))
    
    # Test 3: PNDM with 50 steps (baseline)
    print("\n" + "="*50)
    print("Test 3: PNDM 50 steps (baseline)")
    print("="*50)
    pipe.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    img3, t3, m3 = benchmark_sampler(pipe, prompt, 50, "PNDM")
    results.append(("pndm_50.png", img3))
    
    # Save results
    print("\n" + "="*50)
    print("Saving results...")
    for filename, image in results:
        image.save(f"outputs/{filename}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"DDIM 50 steps: {t1:.2f}s, {m1:.2f}GB")
    print(f"DDIM 20 steps: {t2:.2f}s, {m2:.2f}GB (speedup: {t1/t2:.1f}×)")
    print(f"PNDM 50 steps: {t3:.2f}s, {m3:.2f}GB (baseline)")
    print(f"\nDDIM 50 vs PNDM 50 speedup: {t3/t1:.1f}×")
    print(f"Memory reduction: {(m3-m1)/m3*100:.1f}%")


if __name__ == "__main__":
    main()
