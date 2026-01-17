# PVAR (Physical Vision Augment Robotics)

PVAR은 단일 이미지를 기반으로 객체를 3D 공간으로 복원하고, AI를 활용한 물리적 변형 분석 및 새로운 시점(Novel View Synthesis, NVS) 렌더링을 제공하는 프로젝트입니다.

## 🚀 주요 기능

- **객체 분할 (Segmentation)**: Mask R-CNN을 사용하여 이미지 내 특정 객체를 정밀하게 추출합니다.
- **깊이 추정 (Depth Estimation)**: MiDaS 모델을 활용하여 2D 이미지에서 3D 공간 정보를 생성합니다.
- **3D 포인트 클라우드 생성**: 추출된 객체와 깊이 정보를 결합하여 3D 정점(Vertex) 데이터를 형성합니다.
- **AI 기반 물리 변형 분석**: OpenAI/Gemini를 통해 객체의 특성을 분석하고, 이를 바탕으로 자연스러운 기하학적 변형(Deformation)을 적용합니다.
- **NVS 렌더링**: 각도 변화에 따른 가상 뷰를 렌더링하고, 결과물을 `.obj` 3D 모델 파일로 내보낼 수 있습니다.
- **FastAPI 기반 웹 인터페이스**: 사용자 친화적인 웹 환경에서 실시간 이미지 업로드 및 프로세싱 결과를 확인할 수 있습니다.

## 📁 프로젝트 구조

본 프로젝트는 모듈화된 구조로 설계되어 있어 유지보수와 기능 확장이 용이합니다.

```text
PVAR/
├── src/                    # 핵심 소스 코드
│   ├── core/              # 연산 및 물리 엔진 로직
│   │   ├── deformation.py  # 3D 점구름 변형 알고리즘
│   │   └── rendering.py    # 결과 이미지 렌더링 및 저장
│   ├── models/            # AI 및 비전 모델 인터페이스
│   │   ├── ai.py           # LLM 기반 객체 분석 (OpenAI/Gemini)
│   │   ├── depth.py        # MiDaS 깊이 추정 모델
│   │   └── segmentation.py # Mask R-CNN 객체 분할 모델
│   ├── utils/             # 공통 유틸리티
│   │   ├── image.py        # 이미지 처리 및 마스킹 유틸리티
│   │   └── io.py           # 3D 파일(OBJ) 및 데이터 입출력
│   └── config.py          # 모델 경로, 장치(CPU/GPU) 설정 및 전역 상수
├── out/                    # 생성된 결과물(.png, .obj) 저장 디렉토리
├── __main__.py             # FastAPI 백엔드 서버 엔트리포인트
├── index.html              # 웹 대시보드 프론트엔드
├── .env                    # API 키 및 환경 변수 설정
└── README.md               # 프로젝트 안내서
```

*(참고: `nvs_45.py`는 유틸리티 테스트용 파일로 전체 구조에서 독립되어 있습니다.)*

## 🛠 설치 및 실행 방법

### 1. 전제 조건
- Python 3.8 이상
- CUDA 지원 GPU (권장, 없으면 CPU로 구동 가능)
- OpenAI API Key (AI 변형 분석 기능을 위해 필요)

### 2. 의존성 설치
주요 라이브러리를 설치합니다.
```bash
pip install fastapi uvicorn numpy opencv-python torch torchvision python-dotenv openai
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
서버가 시작되면 브라우저에서 `http://localhost:8000`에 접속하여 서비스를 이용할 수 있습니다.

## ⚙️ 작동 원리

1. **Upload**: 사용자가 사진을 업로드하고 대상 객체를 지정합니다.
2. **Analysis**: AI 모델들이 객체를 인식(Mask)하고 거리를 측정(Depth)합니다.
3. **Reconstruction**: 인식된 정보를 바탕으로 중앙 집중형 3D 좌표계를 생성합니다.
4. **Deformation**: LLM이 객체의 재질을 판단하여 적절한 휨(Bend) 정도를 계산합니다.
5. **Output**: 새로운 각도에서의 이미지와 3D 모델 파일을 생성하여 제공합니다.
