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
    """
    Uses GPT-4o to analyze material properties with Chain of Thought reasoning.
    Returns detailed material analysis including internal/external strength.
    """
    base64_image = encode_image(image_bytes)
    
    system_prompt = """당신은 물리학과 재료공학 전문가입니다. 객체를 분석하여 물리적 특성을 단계별로 추론하세요.

다음 단계를 따라 분석하세요:

1. **재질 식별**: 객체가 무엇으로 만들어졌는지 판단 (플라스틱, 금속, 직물, 고무, 종이, 유리, 세라믹, 목재, 유기물 등)

2. **구조 분석**:
   - 내부 강도 (internal_strength): 속이 비어있는지, 꽉 차있는지 (0.0=완전히 빔, 1.0=완전히 참)
   - 외부 강도 (external_strength): 껍질/표면이 얼마나 단단한지 (0.0=매우 약함, 1.0=매우 단단함)

3. **물리적 특성**:
   - 유연성 (flexibility): 얼마나 쉽게 휘어지는지 (0.0=전혀 안 휨, 1.0=매우 잘 휨)
   - 압축 저항 (compression_resistance): 눌렀을 때 얼마나 저항하는지 (0.0=쉽게 찌그러짐, 1.0=전혀 안 찌그러짐)

4. **변형 예측**: 위 분석을 바탕으로 변형 파라미터 결정
   - bend_amount: 휘어지는 정도 (0.0-1.0)
   - squeeze_amount: 양쪽에서 눌리는 정도 (0.0-1.0)
   - surface_irregularity: 표면 불규칙성 (0.0-0.5, 찌그러진 정도)
   - style: "organic" (부드러운 곡선), "sharp" (날카로운 주름), "crumple" (심하게 구겨짐)

JSON 형식으로 반환하세요. reasoning 필드에 한글로 상세한 추론 과정을 작성하세요."""

    user_prompt = f"""다음 객체를 분석해주세요: {lab}

단계별로 추론하고, 각 수치의 근거를 설명하세요."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    
    try:
        if not response.choices:
            return _get_default_analysis()
        
        data = json.loads(response.choices[0].message.content)
        
        # Validate and normalize values
        return {
            "reasoning": data.get("reasoning", "분석 실패"),
            "material_type": data.get("material_type", "알 수 없음"),
            "internal_strength": min(max(float(data.get("internal_strength", 0.5)), 0.0), 1.0),
            "external_strength": min(max(float(data.get("external_strength", 0.5)), 0.0), 1.0),
            "flexibility": min(max(float(data.get("flexibility", 0.5)), 0.0), 1.0),
            "compression_resistance": min(max(float(data.get("compression_resistance", 0.5)), 0.0), 1.0),
            "deformation_params": {
                "bend_amount": min(max(float(data.get("deformation_params", {}).get("bend_amount", 0.5)), 0.0), 1.0),
                "squeeze_amount": min(max(float(data.get("deformation_params", {}).get("squeeze_amount", 0.3)), 0.0), 1.0),
                "surface_irregularity": min(max(float(data.get("deformation_params", {}).get("surface_irregularity", 0.1)), 0.0), 0.5),
                "style": data.get("deformation_params", {}).get("style", "organic") if data.get("deformation_params", {}).get("style") in ["organic", "sharp", "crumple"] else "organic"
            }
        }
    except Exception as e:
        print(f"VLM JSON Parse failed: {e}")
        import traceback
        traceback.print_exc()
        return _get_default_analysis()

def _get_default_analysis():
    """Return default analysis when VLM fails."""
    return {
        "reasoning": "VLM 분석 실패. 기본값 사용.",
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
