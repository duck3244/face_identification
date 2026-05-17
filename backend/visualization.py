from __future__ import annotations

import cv2
import numpy as np

from config import get_logger
from validators import validate_image_input, ImageInput

logger = get_logger(__name__)


def visualize_recognition(
    img: ImageInput,
    face_results: list[dict],
) -> np.ndarray | None:
    """
    얼굴 인식 결과 시각화

    Parameters:
    -----------
    img: str | np.ndarray
        이미지 파일 경로 또는 BGR ndarray
    face_results: list[dict]
        [{'facial_area': {x,y,w,h}, 'matches': [(identity, score), ...]}, ...]

    Returns:
    --------
    np.ndarray | None
        결과가 그려진 RGB 이미지 배열
    """
    validate_image_input(img)

    if isinstance(img, np.ndarray):
        canvas_bgr = img.copy()
    else:
        canvas_bgr = cv2.imread(img)
        if canvas_bgr is None:
            logger.error(f"이미지를 로드할 수 없습니다: {img}")
            return None

    canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    for face in face_results:
        area = face.get("facial_area") or {}
        x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)
        matches = face.get("matches") or []

        if matches:
            identity, score = matches[0]
            label = f"{identity} ({score:.2f})"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (255, 0, 0)

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(canvas, (x, y - text_h - baseline - 5), (x + text_w, y), color, -1)
        cv2.putText(canvas, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return canvas
