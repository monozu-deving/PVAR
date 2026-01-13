import torch
import numpy as np
from src.config import DEVICE

def midas_depth(rgb):
    tr = torch.hub.load("intel-isl/MiDaS", "transforms")
    tf = tr.dpt_transform
    x = tf(rgb).to(DEVICE)
    md = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(DEVICE).eval()
    with torch.no_grad():
        y = md(x)
        h, w = rgb.shape[:2]
        y = torch.nn.functional.interpolate(y.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False).squeeze(1)
    d = y[0].detach().cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return d
