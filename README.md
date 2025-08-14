# Fashion Text-to-Image Generation

## Overview
This project fine-tunes Stable Diffusion with LoRA on a fashion dataset to generate realistic clothing images from text captions.

## Dataset
- **Source:** `tilak1114/deepfashion`  
- **Content:** Images of clothing with captions describing:
  - Sleeve Length, Lower Clothing Length, Socks, Hat, Glasses, Neckwear, etc.
  - Upper/Lower/Outer clothing fabric and color
- **Format:** Each sample includes an image and its corresponding caption

## Model
- **Base Model:** `runwayml/stable-diffusion-v1-5`  
- **Fine-tuning:** LoRA for efficient adaptation on fashion data  
- **Training Setup:**
  - Image resolution: 512×512  
  - Batch size: 1, gradient accumulation: 4  
  - Max training steps: 3000  
  - Learning rate: 1e-4  

## Usage
### Inference
- Load the fine-tuned LoRA weights:  
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to("cpu")

pipe.load_lora_weights("/content/lora-sd-output", weight_name="pytorch_lora_weights.safetensors")

prompt = "A gentleman wears a long-sleeve shirt with mixed patterns."
image = pipe(prompt).images[0]
image.save("fashion_output.png")
