import cv2
import numpy as np

def to_square_png(img_bgr):
    """Makes image square with padding and returns PNG bytes."""
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    res = np.zeros((side, side, 3), dtype=np.uint8)
    dx, dy = (side - w) // 2, (side - h) // 2
    res[dy:dy+h, dx:dx+w] = img_bgr
    _, buf = cv2.imencode(".png", res)
    return buf.tobytes()

def create_refine_mask(img_bgr):
    """Creates a mask where distorted/black areas are transparent (alpha=0)."""
    h, w = img_bgr.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask[gray < 10] = 0
    _, buf = cv2.imencode(".png", mask)
    return buf.tobytes()
