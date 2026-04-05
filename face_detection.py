from __future__ import annotations

import numpy as np
from deepface import DeepFace
from config import DEFAULT_DETECTOR_BACKEND, DEFAULT_TARGET_SIZE, get_logger
from validators import validate_image_path

logger = get_logger(__name__)


def detect_face(img_path: str, detector_backend: str = DEFAULT_DETECTOR_BACKEND) -> dict | None:
    """
    이미지에��� 얼굴 검출

    Parameters:
    -----------
    img_path: str
        얼굴을 검출할 이미지 경로
    detector_backend: str
        사용할 얼굴 검출 백엔드

    Returns:
    --------
    dict | None
        검출된 얼굴 정보 또는 None
    """
    validate_image_path(img_path)
    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend
        )
        if faces:
            return faces[0]
        else:
            logger.warning("얼굴이 감지되지 않았습니다.")
            return None
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"얼굴 검출 중 오류 발생: {e}")
        return None


def extract_face(
    img_path: str,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND
) -> np.ndarray | None:
    """
    이미지에서 얼굴 영역만 추출

    Parameters:
    -----------
    img_path: str
        얼굴을 추출할 이미지 경로
    target_size: tuple[int, int]
        추출된 얼굴 이미지의 크기
    detector_backend: str
        사용할 얼굴 검출 백엔드

    Returns:
    --------
    np.ndarray | None
        추출된 얼굴 이미지 또는 None
    """
    validate_image_path(img_path)
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        if face_objs:
            return face_objs[0]['face']
        else:
            logger.warning("얼굴이 감지되지 않았습니다.")
            return None
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"얼굴 추출 중 오류 발생: {e}")
        return None
