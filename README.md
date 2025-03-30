# DeepFace-FAISS 얼굴 인식 시스템

이 프로젝트는 DeepFace와 FAISS를 활용한 효율적인 얼굴 인식 시스템입니다. CPU에서도 원활하게 실행될 수 있도록 최적화되어 있습니다.

## 주요 기능

- 얼굴 검출 및 추출
- 얼굴 표현(임베딩) 추출
- FAISS 인덱스를 사용한 빠른 얼굴 검색
- 얼굴 데이터베이스 관리
- 인식 결과 시각화

## 파일 구조

- **config.py**: 전체 시스템의 설정 값 및 기본 경로 정의
- **face_detection.py**: 얼굴 검출 및 추출 관련 함수
- **face_representation.py**: 얼굴 표현(임베딩) 추출 함수
- **database.py**: 얼굴 데이터베이스 관리 클래스
- **visualization.py**: 얼굴 인식 결과 시각화 함수
- **face_recognition.py**: 주요 얼굴 인식 시스템 클래스
- **build_database.py**: 얼굴 데이터베이스 구축 스크립트
- **main.py**: 메인 실행 스크립트
- **requirements.txt**: 필요한
 패키지 목록

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 데이터베이스 디렉토리 생성:
```bash
mkdir -p face_database
```

## 사용 방법

### 1. 얼굴 데이터베이스 구축

얼굴 데이터베이스를 생성하려면 `face_database` 디렉토리 아래에 각 인물의 이름으로 폴더를 만들고, 그 안에 해당 인물의 얼굴 이미지를 넣어주세요.

예시 디렉토리 구조:
```
face_database/
├── 홍길동/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 김철수/
│   ├── image1.jpg
│   └── ...
└── ...
```

그 후, 다음 명령어로 데이터베이스를 구축합니다:

```bash
python build_database.py
```

### 2. 얼굴 인식 실행

테스트할 이미지를 `test_image.jpg`로 저장한 후 다음 명령어를 실행하세요:

```bash
python main.py
```

## 설정 변경

기본 설정은 `config.py` 파일에서 변경할 수 있습니다. 다음과 같은 설정을 사용자화할 수 있습니다:

- 얼굴 인식 모델 (VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace)
- 거리 측정 방법 (cosine, euclidean)
- 임계값
- 데이터베이스 경로
- 및 기타 설정...

## 참고사항

- 이 시스템은 CPU에서 원활하게 실행되도록 최적화되어 있습니다.
- 얼굴 인식 성능을 높이려면 각 인물당 다양한 각도와 표정의 얼굴 이미지를 추가하는 것이 좋습니다.
- FAISS 인덱스를 사용하여 대규모 얼굴 데이터베이스에서도 빠른 검색이 가능합니다.
