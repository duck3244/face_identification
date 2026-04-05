from __future__ import annotations

import cv2
import numpy as np

from deepface import DeepFace
from config import DEFAULT_DETECTOR_BACKEND, get_logger
from validators import validate_image_path

logger = get_logger(__name__)


def visualize_recognition(
    img_path: str,
    recognition_results: list[tuple[str, float]],
    detector_backend: str = DEFAULT_DETECTOR_BACKEND
) -> np.ndarray | None:
    """
    얼굴 인식 결과 시각화

    Parameters:
    -----------
    img_path: str
        인식할 얼굴 이미지 경로
    recognition_results: list[tuple[str, float]]
        [(identity, similarity_score), ...] 형식의 인식 결과 목록
    detector_backend: str
        사용할 얼굴 검출 백엔드

    Returns:
    --------
    np.ndarray | None
        결과가 그려진 RGB 이미지 배열
    """
    validate_image_path(img_path)

    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"이미지를 로드할 수 없습니다: {img_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not recognition_results:
        return img

    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        for i, face_obj in enumerate(faces):
            if i >= len(recognition_results):
                break

            area = face_obj['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            identity, score = recognition_results[i]

            # 바운딩 박스
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 라벨 배경
            label = f"{identity} ({score:.2f})"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x, y - text_h - baseline - 5), (x + text_w, y), (0, 255, 0), -1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img
    except (ValueError, RuntimeError) as e:
        logger.error(f"시각화 중 오류 발생: {e}")
        return img
