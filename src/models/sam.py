import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from src.config import SAM_CHECKPOINT, DEVICE
import os

_sam_predictor = None

def get_sam_predictor():
    """Lazy load SAM model to save memory."""
    global _sam_predictor
    if _sam_predictor is None:
        if not os.path.exists(SAM_CHECKPOINT):
            raise FileNotFoundError(
                f"SAM checkpoint not found at {SAM_CHECKPOINT}. "
                f"Please run 'python download_sam.py' first."
            )
        print(f"Loading SAM model from {SAM_CHECKPOINT}...")
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        sam.to(device=DEVICE)
        _sam_predictor = SamPredictor(sam)
        print("SAM model loaded successfully.")
    return _sam_predictor

def sam_refine_mask(rgb, bbox, initial_mask=None):
    """
    Refine mask using SAM with bounding box prompt.
    
    Args:
        rgb: RGB image (H, W, 3) numpy array
        bbox: Bounding box [x1, y1, x2, y2]
        initial_mask: Optional initial mask from Mask R-CNN (not used, for compatibility)
    
    Returns:
        Refined binary mask (H, W) as uint8
    """
    try:
        predictor = get_sam_predictor()
        predictor.set_image(rgb)
        
        # Convert bbox to SAM format [x1, y1, x2, y2]
        input_box = np.array(bbox)
        
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        # Return the best mask
        refined_mask = masks[0].astype(np.uint8)
        return refined_mask
        
    except Exception as e:
        print(f"SAM refinement failed: {e}")
        # Fallback to initial mask if available
        if initial_mask is not None:
            return initial_mask
        # Otherwise return empty mask
        return np.zeros(rgb.shape[:2], dtype=np.uint8)
