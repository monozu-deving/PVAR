import numpy as np

def deform_points(pts, amount=0.5, style="organic", irregularity=0.0):
    """Deforms the point cloud realistically based on style and irregularity."""
    if pts.shape[0] == 0:
        return pts
    pts_bent = pts.copy()
    
    xs = pts[:, 0]
    xmin, xmax = xs.min(), xs.max()
    x_mid = (xmin + xmax) / 2
    x_range = (xmax - xmin) / 2 + 1e-9
    
    # Normalize X to [-1, 1]
    xn = (xs - x_mid) / x_range
    
    # 1. Base Deformation Shape
    if style == "sharp":
        # Linear V-shape for rigid objects that bend sharply
        bulge = (1.0 - np.abs(xn))
    else:
        # Default quadratic bulge for most objects
        bulge = (1.0 - xn**2)
        
    # Apply Depth Bulge
    pts_bent[:, 2] -= (amount * x_range * bulge)
    
    # 2. Add Irregularity/Noise (e.g. for crumpled cans)
    if irregularity > 0:
        noise_scale = irregularity * x_range * 0.5
        noise = np.random.normal(0, noise_scale, size=pts.shape[0])
        pts_bent[:, 2] += noise
        
    return pts_bent
