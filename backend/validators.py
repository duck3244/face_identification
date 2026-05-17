from __future__ import annotations

import os
from typing import Union

import numpy as np

SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

ImageInput = Union[str, np.ndarray]


def validate_image_path(img_path: str) -> None:
    """이미지 경로 유효성 검증"""
    if not isinstance(img_path, str):
        raise TypeError(f"img_path는 문자열이어야 합니다. 받은 타입: {type(img_path)}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")
    ext = os.path.splitext(img_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"지원하지 않는 이미지 형식입니다: {ext}")


def validate_image_input(img: ImageInput) -> None:
    """이미지 입력 유효성 검증 (path 또는 BGR ndarray 모두 허용)"""
    if isinstance(img, np.ndarray):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"ndarray 이미지는 (H, W, 3) BGR 형식이어야 합니다. 받은 shape: {img.shape}")
        return
    validate_image_path(img)


def validate_threshold(threshold: float) -> None:
    """임계값 유효성 검증"""
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold는 숫자여야 합니다. 받은 타입: {type(threshold)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold는 0.0~1.0 범위여야 합니다. 받은 값: {threshold}")


def validate_top_k(top_k: int) -> None:
    """top_k 유효성 검증"""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k는 1 이상의 정수여야 합니다. 받은 값: {top_k!r}")


def validate_identity(identity: str) -> None:
    """identity 유효성 검증"""
    if not isinstance(identity, str):
        raise TypeError(f"identity는 문자열이어야 합니다. 받은 타입: {type(identity)}")
    if not identity.strip():
        raise ValueError("identity는 빈 문자열일 수 없습니다.")
