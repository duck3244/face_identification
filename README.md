# DeepFace-FAISS 얼굴 인식 시스템

DeepFace와 FAISS를 활용한 효율적인 얼굴 인식 시스템입니다. CPU/GPU 환경 모두 지원하며, 환경변수 및 CLI 인자를 통해 유연하게 설정할 수 있습니다.

## 주요 기능

- 얼굴 검출 및 추출 (DeepFace 기반)
- 얼굴 표현(임베딩) 추출 (VGG-Face, Facenet, ArcFace 등)
- FAISS 인덱스를 사용한 빠른 유사 얼굴 검색
- 얼굴 데이터베이스 관리 (npz + json 안전한 직렬화)
- 인식 결과 시각화
- Gradio 기반 웹 UI (얼굴 인식, DB 관리, 검출/추출, 설정)
- 레거시 pkl 데이터베이스 자동 마이그레이션

## 파일 구조

```
face_identification/
├── config.py              # 시스템 설정 및 로깅 (환경변수 지원)
├── validators.py          # 입력값 검증 (이미지 경로, threshold, top_k)
├── face_detection.py      # 얼굴 검출 및 추출
├── face_representation.py # 얼굴 임베딩 추출
├── database.py            # FAISS 데이터베이스 관리 (지연 인덱스 빌드)
├── face_recognition.py    # 얼굴 인식 시스템 (Facade)
├── visualization.py       # 인식 결과 시각화
├── build_database.py      # 데이터베이스 구축 스크립트
├── main.py                # 메인 실행 스크립트 (argparse 지원)
├── app.py                 # Gradio 웹 UI 애플리케이션
├── requirements.txt       # 의존성 패키지 목록
├── gen_fake_name/         # 테스트용 가짜 이름 생성 유틸리티
└── tests/                 # pytest 기반 자동화 테스트
    ├── conftest.py
    ├── test_validators.py
    ├── test_database.py
    ├── test_face_detection.py
    └── test_build_database.py
```

## 설치 방법

### 요구 사항

- Python >= 3.9
- 권장 환경: conda 가상환경

### 패키지 설치

```bash
pip install -r requirements.txt
```

### 주요 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| deepface | 0.0.93 ~ 0.0.99 | 얼굴 검출, 임베딩 추출 |
| tensorflow | 2.4.0 ~ 2.15.x | 딥러닝 백엔드 |
| faiss-cpu | 1.7.x | 벡터 유사도 검색 |
| opencv-python | >= 4.5.5 | 이미지 처리 |
| numpy | 1.22.0 ~ 1.x | 수치 연산 |
| gradio | >= 4.0.0 | 웹 UI |

> **참고**: `faiss-cpu>=1.8.0`은 `numpy>=2.0`을 요구하여 `tensorflow 2.x`와 충돌합니다. 반드시 `faiss-cpu<1.8.0`을 사용하세요.

## 사용 방법

### 1. 얼굴 데이터베이스 구축

`face_database` 디렉토리 아래에 각 인물의 이름으로 폴더를 만들고, 해당 인물의 얼굴 이미지를 넣어주세요.

```
face_database/
├── Steven_Torres/
│   ├── img25.jpg
│   └── img30.jpg
├── Taylor_Larson/
│   ├── img18.jpg
│   └── img29.jpg
└── Vicki_Stevens/
    ├── img1.jpg
    ├── img2.jpg
    └── img9.jpg
```

데이터베이스 구축:

```bash
python build_database.py
```

### 2. 얼굴 인식 실행

기본 실행:

```bash
python main.py
```

CLI 옵션 사용:

```bash
python main.py --image test_image.jpg --threshold 0.6 --top-k 3
python main.py --model ArcFace --threshold 0.5
python main.py --database my_database.pkl --image query.jpg
```

### 3. 웹 UI 실행 (Gradio)

```bash
python app.py
```

브라우저에서 `http://localhost:7860`으로 접속합니다.

#### 웹 UI 탭 구성

**얼굴 인식 탭**

이미지를 업로드하고 데이터베이스에서 일치하는 인물을 검색합니다.

1. 왼쪽 영역에 이미지를 드래그 앤 드롭하거나 클릭하여 업로드
2. 임계값(0.0~1.0)과 Top-K 값을 조절
3. "인식 실행" 버튼 클릭
4. 오른쪽에 바운딩박스가 표시된 결과 이미지와 매칭된 인물/유사도 테이블 확인

**데이터베이스 관리 탭**

현재 데이터베이스 상태를 확인하고 관리합니다.

- **상태 조회**: 등록된 인물 수, 총 얼굴 수, 인물별 얼굴 수 확인
- **새 얼굴 추가**: 이미지 업로드 + 인물 이름 입력 후 "얼굴 추가" 클릭. 추가 후 자동으로 DB에 저장
- **DB 재구축**: `face_database/` 디렉토리의 인물 폴더 구조를 기반으로 데이터베이스를 처음부터 재구축

**얼굴 검출/추출 탭**

이미지에서 얼굴을 검출하고 크롭된 얼굴 영역을 확인합니다.

1. 이미지 업로드 후 "얼굴 검출" 클릭
2. 바운딩박스가 표시된 이미지, 크롭된 얼굴 이미지, 검출 좌표/신뢰도 확인

**설정 탭**

시스템 설정을 런타임에 변경합니다.

- **인식 모델**: VGG-Face, Facenet, ArcFace 등 선택
- **거리 메트릭**: cosine, euclidean, euclidean_l2
- **검출 백엔드**: opencv, mtcnn, retinaface 등
- **임계값 / Top-K**: 기본값 조절

> 모델을 변경하면 기존 데이터베이스의 임베딩과 호환되지 않습니다. 모델 변경 후 "데이터베이스 관리" 탭에서 DB 재구축이 필요합니다.

#### CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | VGG-Face | 얼굴 인식 모델 |
| `--threshold` | 0.6 | 일치 임계값 (0.0 ~ 1.0) |
| `--top-k` | 3 | 반환할 상위 결과 수 |
| `--image` | test_image.jpg | 테스트 이미지 경로 |
| `--database` | face_database.pkl | 데이터베이스 파일 경로 |

## 설정

### config.py 기본값

| 설정 | 기본값 | 설명 |
|------|--------|------|
| DEFAULT_MODEL_NAME | VGG-Face | 얼굴 인식 모델 |
| DEFAULT_DISTANCE_METRIC | cosine | 거리 측정 방법 |
| DEFAULT_DETECTOR_BACKEND | opencv | 얼굴 검출 백엔드 |
| DEFAULT_THRESHOLD | 0.5 | 일치 임계값 |
| DEFAULT_TOP_K | 3 | 반환할 상위 결과 수 |

### 환경변수

모든 설정은 환경변수로 오버라이드할 수 있습니다.

```bash
export FACE_MODEL=ArcFace           # 인식 모델 변경
export FACE_DISTANCE_METRIC=euclidean  # 거리 메트릭 변경
export FACE_DETECTOR=retinaface     # 검출 백엔드 변경
export FACE_THRESHOLD=0.6           # 임계값 변경
export FACE_TOP_K=5                 # 반환 결과 수 변경
export FACE_USE_GPU=true            # GPU 사용 활성화
```

### 지원 모델 및 백엔드

**얼굴 인식 모델**: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace, GhostFaceNet, Buffalo_L

**얼굴 검출 백엔드**: opencv, ssd, dlib, mtcnn, retinaface, mediapipe, yolov8, yunet, centerface

## 데이터베이스 형식

### 현재 형식 (npz + json)

임베딩은 `face_database.npz`에, 메타데이터는 `face_database.json`에 저장됩니다.

```json
{
  "identities": ["Steven_Torres", "Steven_Torres", "Taylor_Larson"],
  "model_name": "VGG-Face",
  "distance_metric": "cosine"
}
```

### 레거시 pkl 마이그레이션

기존 `.pkl` 파일이 감지되면 자동으로 새 형식(npz + json)으로 마이그레이션됩니다. 별도의 수동 작업이 필요하지 않습니다.

## 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 개별 모듈 테스트
pytest tests/test_validators.py -v
pytest tests/test_database.py -v
pytest tests/test_face_detection.py -v
pytest tests/test_build_database.py -v
```

## 참고사항

- 기본적으로 CPU 모드로 실행됩니다. GPU를 사용하려면 `FACE_USE_GPU=true`를 설정하세요.
- 얼굴 인식 성능을 높이려면 각 인물당 다양한 각도와 표정의 얼굴 이미지를 추가하는 것이 좋습니다.
- FAISS 인덱스 빌드는 지연 방식으로 동작하여, 여러 얼굴을 추가한 후 검색 시점에 한 번만 빌드됩니다.
- 로깅은 Python `logging` 모듈을 사용하며, 로그 레벨은 `config.py`에서 조정할 수 있습니다.

## 라이선스

MIT License
