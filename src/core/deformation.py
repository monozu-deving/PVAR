import numpy as np

def deform_points(pts, 
                  bend_amount=0.5, 
                  squeeze_amount=0.3,
                  internal_strength=0.5,
                  external_strength=0.7,
                  surface_irregularity=0.1,
                  style="organic"):
    """
    Advanced deformation based on material properties.
    
    Args:
        bend_amount: 휘어지는 정도 (0.0-1.0)
        squeeze_amount: 양쪽에서 눌리는 정도 (0.0-1.0)
        internal_strength: 내부 강도 (높을수록 중심부가 덜 변형)
        external_strength: 외부 강도 (높을수록 표면이 덜 변형)
        surface_irregularity: 표면 불규칙성 (0.0-0.5)
        style: 변형 스타일 ("organic", "sharp", "crumple")
    """
    if pts.shape[0] == 0:
        return pts
    
    pts_deformed = pts.copy()
    
    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]
    
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    zmin, zmax = zs.min(), zs.max()
    
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2
    z_mid = (zmin + zmax) / 2
    
    x_range = (xmax - xmin) / 2 + 1e-9
    y_range = (ymax - ymin) / 2 + 1e-9
    z_range = (zmax - zmin) / 2 + 1e-9
    
    # Normalize coordinates to [-1, 1]
    xn = (xs - x_mid) / x_range
    yn = (ys - y_mid) / y_range
    zn = (zs - z_mid) / z_range
    
    # 1. BEND DEFORMATION (휘어짐)
    # External strength reduces bending effect
    bend_factor = bend_amount * (1.0 - external_strength * 0.5)
    
    if style == "sharp":
        # Linear V-shape for rigid objects
        bulge = (1.0 - np.abs(xn))
    else:
        # Quadratic bulge for organic objects
        bulge = (1.0 - xn**2)
    
    # Internal strength protects center from bending
    internal_protection = 1.0 - internal_strength * (1.0 - np.abs(zn))
    bend_effect = bend_factor * x_range * bulge * internal_protection
    pts_deformed[:, 2] -= bend_effect
    
    # 2. SQUEEZE DEFORMATION (압착)
    # Compress from both sides toward center
    squeeze_factor = squeeze_amount * (1.0 - external_strength * 0.3)
    
    # Distance from center in XY plane
    radial_dist = np.sqrt(xn**2 + yn**2)
    
    # Squeeze effect: stronger at edges, weaker at center
    squeeze_profile = radial_dist * (1.0 - internal_strength * 0.5)
    
    # Apply squeeze (move points toward center)
    squeeze_x = -xn * squeeze_factor * x_range * squeeze_profile
    squeeze_y = -yn * squeeze_factor * y_range * squeeze_profile
    
    pts_deformed[:, 0] += squeeze_x
    pts_deformed[:, 1] += squeeze_y
    
    # 3. SURFACE IRREGULARITY (표면 불규칙성)
    if surface_irregularity > 0:
        # External strength reduces surface irregularity
        noise_scale = surface_irregularity * x_range * (1.0 - external_strength * 0.7)
        
        if style == "crumple":
            # Sharp, localized crumples
            noise_x = np.random.normal(0, noise_scale * 0.5, size=pts.shape[0])
            noise_y = np.random.normal(0, noise_scale * 0.5, size=pts.shape[0])
            noise_z = np.random.normal(0, noise_scale, size=pts.shape[0])
            
            pts_deformed[:, 0] += noise_x
            pts_deformed[:, 1] += noise_y
            pts_deformed[:, 2] += noise_z
        else:
            # Smooth surface variations
            noise_z = np.random.normal(0, noise_scale * 0.5, size=pts.shape[0])
            pts_deformed[:, 2] += noise_z
    
    return pts_deformed
