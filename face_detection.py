from deepface import DeepFace
from config import DEFAULT_DETECTOR_BACKEND, DEFAULT_TARGET_SIZE


def detect_face(img_path, detector_backend=DEFAULT_DETECTOR_BACKEND):
    """
    이미지에서 얼굴 검출

    Parameters:
    -----------
    img_path: str
        얼굴을 검출할 이미지 경로
    detector_backend: str
        사용할 얼굴 검출 백엔드

    Returns:
    --------
    dict
        검출된 얼굴 정보
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend
        )
        if faces and len(faces) > 0:
            return faces[0]
        else:
            print("얼굴이 감지되지 않았습니다.")
            return None
    except Exception as e:
        print(f"얼굴 검출 중 오류 발생: {e}")
        return None


def extract_face(img_path, target_size=DEFAULT_TARGET_SIZE, detector_backend=DEFAULT_DETECTOR_BACKEND):
    """
    이미지에서 얼굴 영역만 추출

    Parameters:
    -----------
    img_path: str
        얼굴을 추출할 이미지 경로
    target_size: tuple
        추출된 얼굴 이미지의 크기
    detector_backend: str
        사용할 얼굴 검출 백엔드

    Returns:
    --------
    numpy.ndarray
        추출된 얼굴 이미지
    """
    try:
        # 얼굴 감지 (CPU 모드)
        face_objs = DeepFace.extract_faces(
            img_path=img_path,
            target_size=target_size,
            detector_backend=detector_backend,
            enforce_detection=False  # 얼굴이 명확하지 않은 경우에도 최선을 다해 추출
        )

        if len(face_objs) > 0:
            return face_objs[0]['face']  # 첫 번째 검출된 얼굴 반환
        else:
            print("얼굴이 감지되지 않았습니다.")
            return None
    except Exception as e:
        print(f"얼굴 추출 중 오류 발생: {e}")
        return None

