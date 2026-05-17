from __future__ import annotations

import base64
from collections import Counter
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from api.schemas import DatabaseStatus, PersonStat

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5MB


def read_upload_as_bgr(upload: UploadFile) -> np.ndarray:
    """업로드 파일을 BGR ndarray로 디코딩 (크기 제한 + 검증)"""
    data = upload.file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다 (최대 {MAX_UPLOAD_BYTES // (1024*1024)}MB)")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다. (지원 형식: PNG/JPG/JPEG/BMP/TIFF)")
    return img


def encode_rgb_to_png_b64(rgb_img: np.ndarray) -> str:
    """RGB ndarray → PNG → base64 (브라우저 <img src> 호환)"""
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_bgr_to_png_b64(bgr_img: np.ndarray) -> str:
    """이미 BGR인 ndarray를 PNG base64로 인코딩"""
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def db_status_from_system(system: Any) -> DatabaseStatus:
    """FaceRecognitionSystem → DatabaseStatus 스키마 변환"""
    identities = system.database.identities
    counts = Counter(identities)
    persons = [PersonStat(name=name, face_count=count) for name, count in sorted(counts.items())]
    return DatabaseStatus(
        persons=persons,
        person_count=len(counts),
        total_faces=len(identities),
    )
