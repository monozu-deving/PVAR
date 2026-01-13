import os
import uvicorn
import math
import cv2
import uuid
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv, find_dotenv

# Modular Imports
from src.config import OUT_DIR
from src.models.segmentation import maskrcnn_mask
from src.models.depth import midas_depth
from src.models.ai import get_openai_client, analyze_deformation
from src.core.rendering import render_and_save
from src.core.deformation import deform_points
from src.utils.io import save_as_obj
from src.utils.image import to_square_png, create_refine_mask

# Load .env file explicitly
load_dotenv(find_dotenv())

app = FastAPI()

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)
app.mount("/out", StaticFiles(directory=OUT_DIR), name="out")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/nvs")
async def process_nvs(
    image: UploadFile = File(...),
    target: Optional[str] = Form(None),
    angle: float = Form(45.0),
    refine: bool = Form(False),
    prompt: Optional[str] = Form(None)
):
    try:
        # 1. Load and Mask
        img_content = await image.read()
        nparr = np.frombuffer(img_content, np.uint8)
        rgb_orig = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = rgb_orig.shape[:2]

        print(f"Detecting mask for target: {target or 'largest object'}...")
        mask, box, lab = maskrcnn_mask(rgb_orig, target=target)
        if mask is None:
            raise HTTPException(status_code=400, detail="Object not detected. Try another image.")

        # 2. Depth
        depth = midas_depth(rgb_orig)
        
        # 3. 3D Points
        f = 0.5 * w / math.tan(0.5 * math.radians(60.0))
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        ys, xs = np.where(mask > 0)
        sel = np.arange(xs.size)[::2]
        xs, ys = xs[sel], ys[sel]
        z = depth[ys, xs]
        X, Y = (xs - cx) * z / f, (ys - cy) * z / f
        pts = np.stack([X, Y, z], axis=1).astype(np.float32)
        col = rgb_orig[ys, xs]

        # 4. Center and Smear Removal
        z0 = float(np.median(z))
        depth_threshold = 0.3
        valid_mask = (z >= z0 * (1.0 - depth_threshold)) & (z <= z0 * (1.2 + depth_threshold))
        
        pts = pts[valid_mask]
        col = col[valid_mask]
        z_filtered = z[valid_mask]
        xs_filtered = xs[valid_mask]
        ys_filtered = ys[valid_mask]

        z0_new = float(np.median(z_filtered))
        tx = float((np.mean(xs_filtered) - cx) * z0_new / f)
        ty = float((np.mean(ys_filtered) - cy) * z0_new / f)
        t0 = np.array([tx, ty, z0_new], dtype=np.float32)
        pts_centered = pts - t0

        # 5. Render Normal
        uid = str(uuid.uuid4())[:8]
        print("Cleaning background with median color...")
        bg_mask = (mask == 0)
        median_color = np.median(rgb_orig[bg_mask], axis=0).astype(np.uint8) if np.any(bg_mask) else np.array([128,128,128], dtype=np.uint8)
        cleaned_bg = rgb_orig.copy()
        cleaned_bg[mask > 0] = median_color
        
        files_normal = render_and_save(pts_centered, col, w, h, f, cx, cy, t0, angle, uid, bg_image=cleaned_bg)
        obj_normal = f"{uid}_model.obj"
        save_as_obj(pts_centered, col, os.path.join(OUT_DIR, obj_normal))

        # 6. Render Bent
        client = get_openai_client()
        def_params = {"weight": 0.5, "style": "organic", "irregularity": 0.1}
        if client:
            try:
                print("Analyzing deformation properties...")
                def_params = await analyze_deformation(client, img_content, lab)
                print(f"AI Deformation Params: {def_params}")
            except Exception as e:
                print(f"Deformation analysis failed: {e}")

        pts_bent = deform_points(pts_centered, 
                                 amount=def_params["weight"], 
                                 style=def_params["style"], 
                                 irregularity=def_params["irregularity"])
        
        uid_bent = f"{uid}_bent"
        files_bent = render_and_save(pts_bent, col, w, h, f, cx, cy, t0, angle, uid_bent, bg_image=cleaned_bg)
        obj_bent = f"{uid_bent}.obj"
        save_as_obj(pts_bent, col, os.path.join(OUT_DIR, obj_bent))

        response_data = {
            "status": "success",
            "files": {
                "left": f"/out/{files_normal[0]}",
                "right": f"/out/{files_normal[1]}",
                "obj": f"/out/{obj_normal}",
                "bent_left": f"/out/{files_bent[0]}",
                "bent_right": f"/out/{files_bent[1]}",
                "bent_obj": f"/out/{obj_bent}"
            },
            "label": lab,
            "flexibility": def_params["weight"],
            "def_style": def_params["style"],
            "def_irregularity": def_params["irregularity"],
            "angle": angle
        }
        
        # 7. DALL-E Edit Refinement (Optional)
        if refine and client:
            # We refine the "right" view as an example for now
            target_view_path = os.path.join(OUT_DIR, files_normal[1])
            target_bgr = cv2.imread(target_view_path)
            
            img_png = to_square_png(target_bgr)
            mask_png = create_refine_mask(target_bgr)
            
            refine_prompt = prompt if (prompt and prompt.strip()) else f"A high quality professional photo of a {lab} from a side perspective."
            
            try:
                response = client.images.edit(
                    model="dall-e-2",
                    image=img_png,
                    mask=mask_png,
                    prompt=refine_prompt,
                    n=1,
                    size="1024x1024"
                )
                response_data["refine_url"] = response.data[0].url
            except Exception as e:
                print(f"DALL-E Refinement failed: {e}")

        return response_data

    except Exception as e:
        print(f"Error in process_nvs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
