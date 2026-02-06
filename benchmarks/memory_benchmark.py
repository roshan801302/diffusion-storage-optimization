"""
Comprehensive Memory Benchmark
Test different optimization strategies and measure memory usage
"""

import torch
import time
from diffusers import StableDiffusionPipeline, DDIMScheduler
import gc


def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_config(config_name, pipe_fn, prompt, num_steps=50):
    """Benchmark a specific configuration"""
    clear_memory()
    
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    
    try:
        # Create pipeline
        pipe = pipe_fn()
        
        # Warmup
        _ = pipe(prompt, num_inference_steps=5, output_type="latent")
        clear_memory()
        
        # Actual benchmark
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated()
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        elapsed = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        avg_mem = (torch.cuda.memory_allocated() - start_mem) / 1e9
        
        print(f"✓ Success")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Peak Memory: {peak_mem:.2f} GB")
        print(f"  Avg Memory: {avg_mem:.2f} GB")
        
        # Cleanup
        del pipe
        clear_memory()
        
        return {
            "success": True,
            "time": elapsed,
            "peak_memory": peak_mem,
            "avg_memory": avg_mem
        }
        
    except RuntimeError as e:
        print(f"✗ Failed: {str(e)}")
        clear_memory()
        return {"success": False, "error": str(e)}


def main():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    prompt = "a beautiful landscape with mountains and a lake"
    
    results = {}
    
    # Config 1: Baseline (FP32, no optimizations)
    def baseline():
        return StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None
        ).to("cuda")
    
    # Uncomment to test (requires lots of memory)
    # results["Baseline (FP32)"] = benchmark_config("Baseline (FP32)", baseline, prompt)
    
    # Config 2: FP16 only
    def fp16_only():
        return StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")
    
    results["FP16 Only"] = benchmark_config("FP16 Only", fp16_only, prompt)
    
    # Config 3: FP16 + DDIM
    def fp16_ddim():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
            safety_checker=None
        ).to("cuda")
        return pipe
    
    results["FP16 + DDIM"] = benchmark_config("FP16 + DDIM", fp16_ddim, prompt)
    
    # Config 4: FP16 + DDIM + Attention Slicing
    def fp16_ddim_slicing():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
            safety_checker=None
        ).to("cuda")
        pipe.enable_attention_slicing()
        return pipe
    
    results["FP16 + DDIM + Slicing"] = benchmark_config(
        "FP16 + DDIM + Attention Slicing", fp16_ddim_slicing, prompt
    )
    
    # Config 5: FP16 + DDIM + xFormers
    def fp16_ddim_xformers():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
            safety_checker=None
        ).to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            print("  (xFormers not available, skipping)")
        return pipe
    
    results["FP16 + DDIM + xFormers"] = benchmark_config(
        "FP16 + DDIM + xFormers", fp16_ddim_xformers, prompt
    )
    
    # Config 6: All optimizations
    def all_optimizations():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
            safety_checker=None
        ).to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        return pipe
    
    results["All Optimizations"] = benchmark_config(
        "All Optimizations", all_optimizations, prompt
    )
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Time (s)':<12} {'Peak Mem (GB)':<15} {'Speedup':<10}")
    print("-"*80)
    
    baseline_time = None
    for config, result in results.items():
        if result["success"]:
            if baseline_time is None:
                baseline_time = result["time"]
                speedup = "1.0×"
            else:
                speedup = f"{baseline_time / result['time']:.1f}×"
            
            print(f"{config:<30} {result['time']:<12.2f} {result['peak_memory']:<15.2f} {speedup:<10}")
        else:
            print(f"{config:<30} {'FAILED':<12} {'-':<15} {'-':<10}")


if __name__ == "__main__":
    main()
