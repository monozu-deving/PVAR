import os
import math
import cv2
import numpy as np
from src.config import OUT_DIR

def render_and_save(pts_centered, col, w, h, f, cx, cy, t0, angle_deg, prefix, bg_image=None):
    """Renders 45-degree side views from the point cloud and composites onto background."""
    filenames = []
    
    # Resize background to match input size if provided
    if bg_image is not None:
        bg_image = cv2.resize(bg_image, (w, h))

    for sgn in [-1.0, 1.0]:
        ang = math.radians(angle_deg * sgn)
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        p2 = (pts_centered @ R.T) + t0
        
        img = np.zeros((h, w, 3), dtype=np.uint8)
        mask_rendered = np.zeros((h, w), dtype=np.uint8)
        zbuf = np.full((h, w), np.inf, dtype=np.float32)
        
        X, Y, Z = p2[:, 0], p2[:, 1], p2[:, 2]
        mask_v = Z > 0.1
        X, Y, Z, C = X[mask_v], Y[mask_v], Z[mask_v], col[mask_v]
        
        u = (f * X / Z + cx).astype(np.int32)
        v = (f * Y / Z + cy).astype(np.int32)
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, Z, C = u[valid], v[valid], Z[valid], C[valid]

        for i in range(len(u)):
            if Z[i] < zbuf[v[i], u[i]]:
                zbuf[v[i]:v[i]+1, u[i]:u[i]+1] = Z[i]
                cv2.circle(img, (u[i], v[i]), 2, (int(C[i, 0]), int(C[i, 1]), int(C[i, 2])), -1)
                cv2.circle(mask_rendered, (u[i], v[i]), 2, 255, -1)
        
        # Post-process to fill tiny gaps and smooth (Surface Filling)
        # 1. Close holes in the mask
        kernel_dil = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask_rendered, kernel_dil, iterations=1)
        
        kernel_close = np.ones((7, 7), np.uint8)
        mask_dense = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel_close)
        
        # 2. Smooth the image to get "average color" on the surface
        img_smooth = cv2.medianBlur(img, 5) 
        
        # Composite
        if bg_image is not None:
            bg = bg_image.copy()
            obj_mask = mask_dense > 0
            bg[obj_mask] = img_smooth[obj_mask]
            final_res = bg
        else:
            final_res = img_smooth
            
        side = "left" if sgn < 0 else "right"
        fname = f"{prefix}_{side}_{int(angle_deg)}deg.png"
        cv2.imwrite(os.path.join(OUT_DIR, fname), cv2.cvtColor(final_res, cv2.COLOR_RGB2BGR))
        filenames.append(fname)
    return filenames
