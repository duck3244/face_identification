from __future__ import annotations

import numpy as np
from deepface import DeepFace
from config import DEFAULT_DETECTOR_BACKEND, DEFAULT_TARGET_SIZE, get_logger
from validators import validate_image_input, ImageInput

logger = get_logger(__name__)


def detect_face(img: ImageInput, detector_backend: str = DEFAULT_DETECTOR_BACKEND) -> dict | None:
    """이미지에서 얼굴 검출 (path 또는 BGR ndarray)"""
    validate_image_input(img)
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend=detector_backend,
            enforce_detection=False,
        )
        if faces:
            return faces[0]
        logger.warning("얼굴이 감지되지 않았습니다.")
        return None
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"얼굴 검출 중 오류 발생: {e}")
        return None


def extract_face(
    img: ImageInput,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND
) -> np.ndarray | None:
    """이미지에서 얼굴 영역만 추출 (path 또는 BGR ndarray)"""
    validate_image_input(img)
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        if face_objs:
            return face_objs[0]['face']
        logger.warning("얼굴이 감지되지 않았습니다.")
        return None
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"얼굴 추출 중 오류 발생: {e}")
        return None
