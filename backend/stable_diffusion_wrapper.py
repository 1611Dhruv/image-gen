from diffusers import StableDiffusionPipeline
import torch

class StableDiffusionWrapper:
    def __init__(self) -> None:
        model_id = "prompthero/openjourney"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = pipe.to("cuda")

            
    def generate_images(self, text_prompt: str, num_images: int, steps: int):
        prompt = [text_prompt] * num_images
        images = self.pipe(prompt, num_inference_steps=steps).images
        return images
