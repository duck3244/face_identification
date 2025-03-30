from deepface import DeepFace
from config import DEFAULT_MODEL_NAME, DEFAULT_DETECTOR_BACKEND


def represent_face(img_path, model_name=DEFAULT_MODEL_NAME, detector_backend=DEFAULT_DETECTOR_BACKEND):
    """
    이미지에서 얼굴 표현(임베딩) 추출

    Parameters:
    -----------
    img_path: str
        얼굴 표현을 추출할 이미지 경로
    model_name: str
        얼굴 표현을 추출하는 데 사용할 모델
    detector_backend: str
        사용할 얼굴 검출 백엔드

    Returns:
    --------
    numpy.ndarray
        얼굴 임베딩 벡터
    """
    try:
        # CPU 모드 명시적으로 지정
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            enforce_detection=False,  # 얼굴 감지 강제 비활성화
            detector_backend=detector_backend  # 더 가벼운 OpenCV 백엔드 사용
        )

        if embedding_objs:
            return embedding_objs[0]["embedding"]  # 첫 번째 얼굴의 임베딩 반환
        else:
            print("얼굴 임베딩을 추출할 수 없습니다.")
            return None
    except Exception as e:
        print(f"얼굴 표현 추출 중 오류 발생: {e}")
        return None

