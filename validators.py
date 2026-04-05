from __future__ import annotations

import os

SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


def validate_image_path(img_path: str) -> None:
    """이미지 경로 유효성 검증"""
    if not isinstance(img_path, str):
        raise TypeError(f"img_path는 문자열이어야 합니다. 받은 타입: {type(img_path)}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")
    ext = os.path.splitext(img_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"지원하지 않는 이미지 형식입니다: {ext}")


def validate_threshold(threshold: float) -> None:
    """임계값 유효성 검증"""
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold는 숫자여야 합니다. 받은 타입: {type(threshold)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold는 0.0~1.0 범위여야 합니다. 받은 값: {threshold}")


def validate_top_k(top_k: int) -> None:
    """top_k 유효성 검증"""
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k는 1 이상의 정수여야 합니다. 받은 값: {top_k}")
