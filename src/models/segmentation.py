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
