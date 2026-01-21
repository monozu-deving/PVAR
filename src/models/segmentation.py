import torch
import torchvision
import numpy as np
from src.config import DEVICE, COCO

def maskrcnn_mask(rgb, target=None, score_th=0.6):
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = t.to(DEVICE)
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE).eval()
    with torch.no_grad():
        o = m([t])[0]
    scores = o["scores"].detach().cpu().numpy()
    if scores.size == 0: return None, None, None
    keep = np.where(scores >= float(score_th))[0]
    if keep.size == 0: return None, None, None
    labels = o["labels"].detach().cpu().numpy()
    boxes = o["boxes"].detach().cpu().numpy()
    masks = o["masks"].detach().cpu().numpy()[:, 0, :, :]

    if target is not None and target in COCO:
        tid = COCO.index(target)
        idx = keep[labels[keep] == tid]
        if idx.size > 0:
            a = (boxes[idx][:, 2] - boxes[idx][:, 0]) * (boxes[idx][:, 3] - boxes[idx][:, 1])
            j = idx[int(np.argmax(a))]
            return (masks[j] > 0.5).astype(np.uint8), boxes[j], target

    a = (boxes[keep][:, 2] - boxes[keep][:, 0]) * (boxes[keep][:, 3] - boxes[keep][:, 1])
    j = keep[int(np.argmax(a))]
    label_idx = int(labels[j])
    label_name = COCO[label_idx] if label_idx < len(COCO) else "unknown"
    return (masks[j] > 0.5).astype(np.uint8), boxes[j], label_name

def detect_all_objects(rgb, score_th=0.6):
    """
    Detect all objects in the image and return their information.
    
    Args:
        rgb: RGB image (H, W, 3) numpy array
        score_th: Score threshold for detection
    
    Returns:
        List of dicts with object info: [{"id": 0, "label": "person", "score": 0.95, "bbox": [x1,y1,x2,y2], "mask_index": 0}, ...]
    """
    import base64
    import cv2
    from io import BytesIO
    from PIL import Image
    
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = t.to(DEVICE)
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE).eval()
    
    with torch.no_grad():
        o = m([t])[0]
    
    scores = o["scores"].detach().cpu().numpy()
    if scores.size == 0:
        return []
    
    keep = np.where(scores >= float(score_th))[0]
    if keep.size == 0:
        return []
    
    labels = o["labels"].detach().cpu().numpy()
    boxes = o["boxes"].detach().cpu().numpy()
    masks = o["masks"].detach().cpu().numpy()[:, 0, :, :]
    
    objects = []
    for idx in keep:
        bbox = boxes[idx]
        label_idx = int(labels[idx])
        label_name = COCO[label_idx] if label_idx < len(COCO) else "unknown"
        score = float(scores[idx])
        
        # Create thumbnail from bounding box
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        
        # Crop and create thumbnail
        thumbnail = rgb[y1:y2, x1:x2].copy()
        
        # Apply mask to thumbnail for better visualization
        mask = masks[idx]
        mask_crop = mask[y1:y2, x1:x2]
        mask_3ch = np.stack([mask_crop] * 3, axis=-1)
        thumbnail = (thumbnail * (mask_crop[:, :, None] > 0.5)).astype(np.uint8)
        
        # Resize thumbnail to 100x100
        thumbnail_pil = Image.fromarray(thumbnail)
        thumbnail_pil.thumbnail((100, 100), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        thumbnail_pil.save(buffer, format="PNG")
        thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        objects.append({
            "id": int(idx),
            "label": label_name,
            "score": score,
            "bbox": bbox.tolist(),
            "thumbnail": f"data:image/png;base64,{thumbnail_b64}"
        })
    
    return objects
