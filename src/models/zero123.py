import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from src.config import ZERO123_MODEL, DEVICE, USE_LOW_VRAM_MODE
import os

_zero123_pipeline = None

def get_zero123_pipeline():
    """Lazy load Zero123++ pipeline with low VRAM optimizations."""
    global _zero123_pipeline
    if _zero123_pipeline is None:
        print(f"Loading Zero123++ model: {ZERO123_MODEL}...")
        
        # Load pipeline with float16 for memory efficiency
        pipeline = DiffusionPipeline.from_pretrained(
            ZERO123_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        if DEVICE == "cuda" and USE_LOW_VRAM_MODE:
            # Enable memory optimizations for low VRAM GPUs (MX450, etc.)
            print("Enabling low VRAM optimizations...")
            pipeline.enable_sequential_cpu_offload()
            pipeline.enable_vae_slicing()
        else:
            pipeline = pipeline.to(DEVICE)
        
        _zero123_pipeline = pipeline
        print("Zero123++ model loaded successfully.")
    
    return _zero123_pipeline

def generate_multiview(rgb_image, output_dir, uid):
    """
    Generate 6-view images (3x2 grid) from a single RGB image using Zero123++.
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3)
        output_dir: Directory to save generated views
        uid: Unique identifier for output files
    
    Returns:
        List of 6 output file paths
    """
    try:
        pipeline = get_zero123_pipeline()
        
        # Convert numpy to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Resize to 256x256 as required by Zero123++
        pil_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
        
        print("Generating 6 novel views with Zero123++...")
        
        # Generate multi-view image (returns 3x2 grid)
        result = pipeline(
            pil_image,
            num_inference_steps=30,  # Reduced for speed
            guidance_scale=4.0,
        ).images[0]
        
        # Split the 3x2 grid into 6 individual images
        grid_width, grid_height = result.size
        view_width = grid_width // 3
        view_height = grid_height // 2
        
        view_labels = [
            "front_left", "front", "front_right",
            "back_left", "back", "back_right"
        ]
        
        output_paths = []
        
        for idx, label in enumerate(view_labels):
            row = idx // 3
            col = idx % 3
            
            left = col * view_width
            top = row * view_height
            right = left + view_width
            bottom = top + view_height
            
            view_img = result.crop((left, top, right, bottom))
            
            output_path = os.path.join(output_dir, f"{uid}_hifi_{label}.png")
            view_img.save(output_path)
            output_paths.append(f"{uid}_hifi_{label}.png")
        
        print(f"Generated {len(output_paths)} high-fidelity views.")
        return output_paths
        
    except Exception as e:
        print(f"Zero123++ generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []
