import json
import base64
from openai import OpenAI
from src.config import DEVICE

def get_openai_client():
    from dotenv import load_dotenv, find_dotenv
    import os
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

async def analyze_deformation(client, image_bytes, lab):
    """Uses GPT-4o to analyze material and return structured deformation params."""
    base64_image = encode_image(image_bytes)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a material physics expert. Analyze the object and determine its deformation properties. Return ONLY a JSON object with: 'weight' (0.0-1.0), 'style' ('organic', 'sharp', or 'crumple'), and 'irregularity' (0.0-0.5). IMPORTANT: Preserve structural integrity. For 'organic' or smooth objects, use very low 'irregularity' (e.g., 0.01-0.05). High 'irregularity' (>0.2) is ONLY for severely crumpled or fragmented materials."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Analyze the {lab} for realistic physical deformation while maintaining its overall shape."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=100,
        response_format={"type": "json_object"}
    )
    try:
        if not response.choices:
            return {"weight": 0.5, "style": "organic", "irregularity": 0.1}
        data = json.loads(response.choices[0].message.content)
        return {
            "weight": min(max(float(data.get("weight", 0.5)), 0.0), 1.0),
            "style": data.get("style", "organic") if data.get("style") in ["organic", "sharp", "crumple"] else "organic",
            "irregularity": min(max(float(data.get("irregularity", 0.1)), 0.0), 0.5)
        }
    except Exception as e:
        print(f"VLM JSON Parse failed: {e}")
        return {"weight": 0.5, "style": "organic", "irregularity": 0.1}
