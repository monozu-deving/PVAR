import os

def save_as_obj(pts_centered, col, filepath):
    """Saves the given points and colors to an .obj file with Three.js friendly orientation."""
    with open(filepath, "w") as f:
        for i in range(len(pts_centered)):
            x, y, z = pts_centered[i]
            r, g, b = col[i] / 255.0
            f.write(f"v {x} {-y} {-z} {r} {g} {b}\n")
