from __future__ import annotations

from deepface import DeepFace
from config import DEFAULT_MODEL_NAME, DEFAULT_DETECTOR_BACKEND, get_logger
from validators import validate_image_path

logger = get_logger(__name__)


def represent_face(
    img_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND
) -> list[float] | None:
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
    list[float] | None
        얼굴 임베딩 벡터 또는 None
    """
    validate_image_path(img_path)
    try:
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend=detector_backend
        )

        if embedding_objs:
            return embedding_objs[0]["embedding"]
        else:
            logger.warning("얼굴 임베딩을 추출할 수 없습니다.")
            return None
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"얼굴 표현 추출 중 오류 발생: {e}")
        return None
