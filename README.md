# PVAR (Physical Vision Augment Robotics)

PVAR은 단일 이미지를 기반으로 객체를 3D 공간으로 복원하고, AI를 활용한 물리적 변형 분석 및 새로운 시점(Novel View Synthesis, NVS) 렌더링을 제공하는 프로젝트입니다.

## 🚀 주요 기능

- **객체 분할 (Segmentation)**: Mask R-CNN 및 **SAM (Segment Anything)**을 사용하여 정밀한 마스킹을 수행합니다.
- **깊이 추정 (Depth Estimation)**: MiDaS 모델을 활용하여 2D 이미지에서 3D 공간 정보를 생성합니다.
- **High-Fidelity NVS**: **Zero123++**를 통합하여 단일 이미지로부터 일관된 6개 시점(3x2 grid)의 고화질 novel view를 생성합니다.
- **AI 기반 물리 변형 분석**: GPT-4o 전문가 시스템을 통해 객체의 재질(volume, shell, box 등)을 분석하고 현실적인 변형 파라미터를 도출합니다.
- **NVS 렌더링 및 하이브리드 모드**: 초고속 MiDaS 포인트 클라우드 방식과 고품질 Zero123++ 생성 방식을 선택적으로 사용할 수 있습니다.
- **저사양 GPU 최적화**: MX450(2GB VRAM) 등 저사양 환경에서도 구동 가능하도록 Sequential Offloading 및 슬라이싱 기술을 적용했습니다.

## 📁 프로젝트 구조

```text
PVAR/
├── src/                    # 핵심 소스 코드
│   ├── core/              # 연산 및 물리 엔진 로직
│   │   ├── deformation.py  # 3D 점구름 변형 알고리즘
│   │   └── rendering.py    # 결과 이미지 렌더링 (Left/Center/Right)
│   ├── models/            # AI 및 비전 모델 인터페이스
│   │   ├── ai.py           # GPT-4o 기반 물리 속성 분석
│   │   ├── sam.py          # SAM (Segment Anything) 모델
│   │   ├── zero123.py      # Zero123++ 다각도 생성 모델
│   │   ├── depth.py        # MiDaS 깊이 추정
│   │   └── segmentation.py # Mask R-CNN 기본 분할
│   ├── utils/             # 공통 유틸리티
│   │   └── ...
├── weights/                # 모델 가중치 폴더 (SAM 등 저장)
├── out/                    # 생성된 결과물(.png, .obj) 저장 디렉토리
├── __main__.py             # FastAPI 백엔드 서버 엔트리포인트
├── index.html              # 웹 대시보드 프론트엔드
├── download_sam.py         # SAM 가중치 자동 다운로드 스크립트
└── ...
```

*(참고: `nvs_45.py`는 유틸리티 테스트용 파일로 전체 구조에서 독립되어 있습니다.)*

## 🛠 설치 및 실행 방법

### 1. 전제 조건
- Python 3.8 이상
- NVIDIA GPU (MX450 이상 권장, CUDA 지원 필요)
- OpenAI API Key

### 2. 가중치 및 의존성 설치
```bash
# 1. CUDA 지원 PyTorch 설치 (Windows/CUDA 12.4 예시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. 기타 의존성 설치
pip install fastapi uvicorn numpy opencv-python python-dotenv openai segment-anything diffusers accelerate transformers

# 3. SAM 모델 가중치 다운로드
python download_sam.py
```

### 3. 환경 설정
`.env` 파일을 시스템 루트에 생성하고 필요한 정보를 입력합니다.
```env
OPENAI_API_KEY=your_api_key_here
```

### 4. 서버 실행
```bash
python __main__.py
```
접속 주소: `http://localhost:8000`

## ⚙️ 작동 워크플로우

1. **Upload & Settings**: 이미지 업로드 후 대상 객체(target)와 모드(Fast vs High-Fidelity)를 선택합니다.
2. **Hybrid Segmentation**: Mask R-CNN으로 영역을 잡고, 고화질 모드 시 SAM이 정밀하게 경계를 다듬습니다.
3. **Physical Analysis**: GPT-4o가 객체의 시각적 정보를 분석하여 현실적인 굽힘(Bend)과 압착(Grasp) 수치를 제안합니다.
4. **Multi-View Generation**: 
   - **Fast**: MiDaS 깊이 값을 기반으로 포인트 클라우드를 회전시켜 3개 시점 렌더링.
   - **High-Fidelity**: Zero123++이 일관된 6개의 고화질novel view 생성.
5. **Output**: 가공된 이미지와 변형된 3D 모델(.obj)을 즉시 확인합니다.
