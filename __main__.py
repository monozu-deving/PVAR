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
from src.models.segmentation import maskrcnn_mask, detect_all_objects
from src.models.depth import midas_depth
from src.models.ai import get_openai_client, analyze_deformation
from src.models.sam import sam_refine_mask
from src.models.zero123 import generate_multiview
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

# Store detection results temporarily (in production, use Redis or similar)
_detection_cache = {}

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    """Detect all objects in the uploaded image."""
    try:
        img_content = await image.read()
        nparr = np.frombuffer(img_content, np.uint8)
        rgb_orig = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        print("Detecting all objects in image...")
        objects = detect_all_objects(rgb_orig, score_th=0.6)
        
        # Cache the image and detection results
        import hashlib
        img_hash = hashlib.md5(img_content).hexdigest()
        _detection_cache[img_hash] = {
            "image": rgb_orig,
            "objects": objects
        }
        
        return {
            "status": "success",
            "image_hash": img_hash,
            "objects": objects,
            "count": len(objects)
        }
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nvs")
async def process_nvs(
    image: UploadFile = File(...),
    target: Optional[str] = Form(None),
    object_id: Optional[int] = Form(None),
    angle: float = Form(45.0),
    mode: str = Form("fast"),  # "fast" or "hifi"
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
        
        # If object_id is provided, use cached detection results
        if object_id is not None:
            # Try to find in cache
            import hashlib
            img_hash = hashlib.md5(img_content).hexdigest()
            if img_hash in _detection_cache:
                cached = _detection_cache[img_hash]
                objects = cached["objects"]
                # Find the object by id
                selected_obj = next((obj for obj in objects if obj["id"] == object_id), None)
                if selected_obj:
                    # Re-run detection to get mask for this specific object
                    mask, box, lab = maskrcnn_mask(rgb_orig, target=selected_obj["label"])
                else:
                    mask, box, lab = maskrcnn_mask(rgb_orig, target=target)
            else:
                mask, box, lab = maskrcnn_mask(rgb_orig, target=target)
        else:
            mask, box, lab = maskrcnn_mask(rgb_orig, target=target)
            
        if mask is None:
            raise HTTPException(status_code=400, detail="Object not detected. Try another image.")
        
        # High-Fidelity Mode: Refine mask with SAM
        if mode == "hifi":
            try:
                print("Refining mask with SAM...")
                mask = sam_refine_mask(rgb_orig, box, mask)
            except Exception as e:
                print(f"SAM refinement failed, using Mask R-CNN result: {e}")

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

        # 5. Render Normal with Gaussian Background
        uid = str(uuid.uuid4())[:8]
        print("Rendering with gaussian noise background...")
        
        files_normal = render_and_save(pts_centered, col, w, h, f, cx, cy, t0, angle, uid, bg_mode="gaussian")
        obj_normal = f"{uid}_model.obj"
        save_as_obj(pts_centered, col, os.path.join(OUT_DIR, obj_normal))

        # 6. AI Material Analysis
        client = get_openai_client()
        analysis = {
            "reasoning": "VLM 분석을 사용할 수 없습니다.",
            "material_type": "알 수 없음",
            "internal_strength": 0.5,
            "external_strength": 0.5,
            "flexibility": 0.5,
            "compression_resistance": 0.5,
            "deformation_params": {
                "bend_amount": 0.5,
                "squeeze_amount": 0.3,
                "surface_irregularity": 0.1,
                "style": "organic"
            }
        }
        
        if client:
            try:
                print("Analyzing material properties with VLM...")
                analysis = await analyze_deformation(client, img_content, lab)
                print(f"VLM Analysis: {analysis['material_type']}")
                print(f"Reasoning: {analysis['reasoning'][:100]}...")  # Print first 100 chars
            except Exception as e:
                print(f"VLM analysis failed: {e}")

        # 7. Render Bent with Advanced Deformation
        pts_bent = deform_points(
            pts_centered,
            bend_amount=analysis["deformation_params"]["bend_amount"],
            squeeze_amount=analysis["deformation_params"]["squeeze_amount"],
            internal_strength=analysis["internal_strength"],
            external_strength=analysis["external_strength"],
            surface_irregularity=analysis["deformation_params"]["surface_irregularity"],
            style=analysis["deformation_params"]["style"]
        )
        
        uid_bent = f"{uid}_bent"
        files_bent = render_and_save(pts_bent, col, w, h, f, cx, cy, t0, angle, uid_bent, bg_mode="gaussian")
        obj_bent = f"{uid_bent}.obj"
        save_as_obj(pts_bent, col, os.path.join(OUT_DIR, obj_bent))

        response_data = {
            "status": "success",
            "mode": mode,
            "files": {
                "center": f"/out/{files_normal[0]}",
                "left": f"/out/{files_normal[1]}",
                "right": f"/out/{files_normal[2]}",
                "obj": f"/out/{obj_normal}",
                "bent_center": f"/out/{files_bent[0]}",
                "bent_left": f"/out/{files_bent[1]}",
                "bent_right": f"/out/{files_bent[2]}",
                "bent_obj": f"/out/{obj_bent}"
            },
            "label": lab,
            "material_analysis": {
                "reasoning": analysis["reasoning"],
                "material_type": analysis["material_type"],
                "internal_strength": analysis["internal_strength"],
                "external_strength": analysis["external_strength"],
                "flexibility": analysis["flexibility"],
                "compression_resistance": analysis["compression_resistance"],
                "bend_amount": analysis["deformation_params"]["bend_amount"],
                "squeeze_amount": analysis["deformation_params"]["squeeze_amount"],
                "surface_irregularity": analysis["deformation_params"]["surface_irregularity"],
                "style": analysis["deformation_params"]["style"]
            },
            "angle": angle
        }
        
        # High-Fidelity Mode: Generate 6-view with Zero123++
        if mode == "hifi":
            try:
                print("Generating high-fidelity multi-view with Zero123++...")
                # Extract masked object for Zero123++
                masked_rgb = rgb_orig.copy()
                masked_rgb[mask == 0] = 255  # White background
                
                hifi_views = generate_multiview(masked_rgb, OUT_DIR, uid)
                response_data["hifi_views"] = [f"/out/{view}" for view in hifi_views]
            except Exception as e:
                print(f"Zero123++ generation failed: {e}")
                response_data["hifi_views"] = []
        
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
