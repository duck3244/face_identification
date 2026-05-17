from __future__ import annotations

from deepface import DeepFace
from config import DEFAULT_MODEL_NAME, DEFAULT_DETECTOR_BACKEND, get_logger
from validators import validate_image_input, ImageInput

logger = get_logger(__name__)


def represent_faces(
    img: ImageInput,
    model_name: str = DEFAULT_MODEL_NAME,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND
) -> list[dict] | None:
    """
    이미지에서 검출된 모든 얼굴의 임베딩과 바운딩 박스 추출

    Parameters:
    -----------
    img: str | np.ndarray
        이미지 파일 경로 또는 BGR ndarray
    """
    validate_image_input(img)
    try:
        embedding_objs = DeepFace.represent(
            img_path=img,
            model_name=model_name,
            enforce_detection=False,
            detector_backend=detector_backend
        )
        if embedding_objs:
            return embedding_objs
        logger.warning("얼굴 임베딩을 추출할 수 없습니다.")
        return None
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"얼굴 표현 추출 중 오류 발생: {e}")
        return None


def represent_face(
    img: ImageInput,
    model_name: str = DEFAULT_MODEL_NAME,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND
) -> list[float] | None:
    """이미지에서 첫 번째 얼굴의 임베딩만 추출 (DB 등록용)"""
    faces = represent_faces(img, model_name, detector_backend)
    return faces[0]["embedding"] if faces else None
