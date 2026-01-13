import os
import math
import argparse
import numpy as np
import cv2
import torch
import torchvision
from PIL import Image

COCO = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def _load_img(p):
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read: {p}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb

def _maskrcnn_mask(rgb, target=None, score_th=0.6, device="cuda"):
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = t.to(device)
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()
    with torch.no_grad():
        o = m([t])[0]
    scores = o["scores"].detach().cpu().numpy()
    if scores.size == 0:
        return None, None, None
    keep = np.where(scores >= float(score_th))[0]
    if keep.size == 0:
        return None, None, None
    labels = o["labels"].detach().cpu().numpy()
    boxes = o["boxes"].detach().cpu().numpy()
    masks = o["masks"].detach().cpu().numpy()[:, 0, :, :]

    if target is not None and target in COCO:
        tid = COCO.index(target)
        idx = keep[labels[keep] == tid]
        if idx.size > 0:
            a = (boxes[idx][:, 2] - boxes[idx][:, 0]) * (boxes[idx][:, 3] - boxes[idx][:, 1])
            j = idx[int(np.argmax(a))]
            mk = masks[j]
            bx = boxes[j]
            return (mk > 0.5).astype(np.uint8), bx, target

    a = (boxes[keep][:, 2] - boxes[keep][:, 0]) * (boxes[keep][:, 3] - boxes[keep][:, 1])
    j = keep[int(np.argmax(a))]
    mk = masks[j]
    bx = boxes[j]
    return (mk > 0.5).astype(np.uint8), bx, COCO[int(labels[j])]

def _grabcut_mask(bgr, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    r = (x1, y1, x2 - x1, y2 - y1)
    m = np.zeros((h, w), np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, m, r, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    out = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return out

def _midas_depth(rgb, device="cuda"):
    tr = torch.hub.load("intel-isl/MiDaS", "transforms")
    tf = tr.dpt_transform
    x = tf(rgb).to(device)
    md = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval()
    with torch.no_grad():
        y = md(x)
        y = torch.nn.functional.interpolate(
            y.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze(1)
    d = y[0].detach().cpu().numpy()
    d = d - d.min()
    if d.max() > 1e-9:
        d = d / d.max()
    d = 1.0 / (d * 0.95 + 0.05)
    return d.astype(np.float32)

def _intrinsics(w, h, fov_deg=60.0):
    f = 0.5 * w / math.tan(0.5 * math.radians(float(fov_deg)))
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    return float(f), float(cx), float(cy)

def _obj_center_from_mask(mask, depth):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    z = float(np.median(depth[ys, xs]))
    return cx, cy, z

def _points_from_mask(rgb, mask, depth, f, cx, cy, step=2, z_scale=1.0):
    h, w = depth.shape
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None, None
    sel = np.arange(xs.size)
    if int(step) > 1:
        sel = sel[::int(step)]
    xs = xs[sel]
    ys = ys[sel]
    z = depth[ys, xs] * float(z_scale)
    X = (xs - cx) * z / f
    Y = (ys - cy) * z / f
    Z = z
    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
    col = rgb[ys, xs].astype(np.uint8)
    return pts, col

def _rot_y(deg):
    a = math.radians(float(deg))
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float32)

def _render_points(pts, col, w, h, f, cx, cy):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]
    ok = Z > 1e-6
    X = X[ok]; Y = Y[ok]; Z = Z[ok]; C = col[ok]
    u = (f * (X / Z) + cx)
    v = (f * (Y / Z) + cy)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui = ui[inb]; vi = vi[inb]; Z = Z[inb]; C = C[inb]
    for i in range(ui.size):
        x = ui[i]; y = vi[i]; z = Z[i]
        if z < zbuf[y, x]:
            zbuf[y, x] = z
            img[y, x] = C[i]
    return img

def _inpaint_holes(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = (g == 0).astype(np.uint8) * 255
    if int(m.sum()) == 0:
        return bgr
    out = cv2.inpaint(bgr, m, 3, cv2.INPAINT_TELEA)
    return out

def _smooth_mask(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return (m > 0).astype(np.uint8)

def _bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x1 = int(xs.min()); x2 = int(xs.max())
    y1 = int(ys.min()); y2 = int(ys.max())
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--target", default=None)
    ap.add_argument("--angle", type=float, default=45.0)
    ap.add_argument("--score", type=float, default=0.6)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fallback_grabcut", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bgr, rgb = _load_img(args.inp)
    h, w = rgb.shape[:2]

    mask, box, lab = _maskrcnn_mask(rgb, target=args.target, score_th=args.score, device=args.device)
    if mask is None:
        if not args.fallback_grabcut:
            raise RuntimeError("maskrcnn failed. try --fallback_grabcut with a manual target or use another image.")
        box = np.array([w*0.2, h*0.2, w*0.8, h*0.8], dtype=np.float32)
        mask = _grabcut_mask(bgr, box)
        lab = "grabcut"

    mask = _smooth_mask(mask)
    if box is None:
        box = _bbox_from_mask(mask)

    depth = _midas_depth(rgb, device=args.device)
    f, cx, cy = _intrinsics(w, h, fov_deg=args.fov)

    c = _obj_center_from_mask(mask, depth)
    if c is None:
        raise RuntimeError("empty mask")
    ocx, ocy, ocz = c

    pts, col = _points_from_mask(rgb, mask, depth, f, cx, cy, step=args.step, z_scale=1.0)
    if pts is None:
        raise RuntimeError("no points")

    z0 = float(np.median(pts[:, 2]))
    t0 = np.array([(ocx - cx) * z0 / f, (ocy - cy) * z0 / f, z0], dtype=np.float32)
    pts = pts - t0

    for sgn in [-1.0, +1.0]:
        ang = float(args.angle) * float(sgn)
        R = _rot_y(ang)
        p2 = (R @ pts.T).T + t0
        r = _render_points(p2, col, w, h, f, cx, cy)
        out = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
        out = _inpaint_holes(out)
        name = f"view_{'left' if sgn<0 else 'right'}_{abs(int(args.angle))}deg_{lab}.png"
        cv2.imwrite(os.path.join(args.outdir, name), out)

    mvis = (mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, f"mask_{lab}.png"), mvis)
    dvis = (depth / (depth.max() + 1e-9) * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, "depth.png"), dvis)

if __name__ == "__main__":
    main()
