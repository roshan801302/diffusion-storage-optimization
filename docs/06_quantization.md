# Quantization and Compression

## Numerical Precision Reduction

Reduce bit-width of model weights and activations to save memory and accelerate inference.

## Precision Options

| Precision | Bits | Memory | Speed | Quality |
|-----------|------|--------|-------|---------|
| FP32 | 32 | 1.0× | 1.0× | Baseline |
| FP16 | 16 | 0.5× | 1.5-2× | Negligible loss |
| BF16 | 16 | 0.5× | 1.5-2× | Better range than FP16 |
| INT8 | 8 | 0.25× | 2-3× | Moderate loss |
| INT4 | 4 | 0.125× | 3-4× | Significant loss |

## FP16 / BF16 Implementation

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
```

## INT8 Quantization

### Post-Training Quantization (PTQ)
Fast quantization without retraining.

```python
import torch
from torch.quantization import quantize_dynamic

# Dynamic quantization (weights only)
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Quantization-Aware Training (QAT)
Train with quantization in the loop for better quality.

```python
import torch.quantization as quant

model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model_prepared = quant.prepare_qat(model)

# Train with quantization simulation
for data in dataloader:
    loss = train_step(model_prepared, data)
    loss.backward()
    optimizer.step()

# Convert to quantized model
model_quantized = quant.convert(model_prepared)
```

## Mixed Precision Strategy

Quantize less sensitive layers more aggressively:

```python
# Keep attention in FP16, quantize FFN to INT8
quantization_config = {
    "attention": "fp16",
    "feed_forward": "int8",
    "conv_layers": "int8",
    "output_layer": "fp16"
}
```

## Per-Channel Quantization

Better accuracy than per-tensor:

```python
# Per-channel scales for each output channel
scales = compute_scales_per_channel(weights)
quantized_weights = quantize_per_channel(weights, scales)
```

## Recommendations

1. **Always use FP16/BF16** - free 2× speedup
2. **INT8 for deployment** - use QAT for best results
3. **Mixed precision** - keep critical layers in higher precision
4. **Calibration dataset** - use representative data for PTQ
