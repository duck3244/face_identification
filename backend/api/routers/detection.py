from __future__ import annotations

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from api.deps import AppState
from api.schemas import DetectResponse, FacialArea
from api.utils import encode_bgr_to_png_b64, read_upload_as_bgr
from face_detection import detect_face, extract_face

router = APIRouter()


def _state(request: Request) -> AppState:
    return request.app.state.app_state


@router.post("/detect", response_model=DetectResponse)
def detect(request: Request, image: UploadFile = File(...)) -> DetectResponse:
    state = _state(request)
    bgr = read_upload_as_bgr(image)

    face_info = detect_face(bgr, detector_backend=state.detector_backend)
    if face_info is None:
        raise HTTPException(status_code=404, detail="얼굴이 검출되지 않았습니다.")

    extracted = extract_face(bgr, detector_backend=state.detector_backend)
    extracted_b64 = None
    if extracted is not None:
        extracted_u8 = (extracted * 255).astype(np.uint8) if extracted.max() <= 1.0 else extracted.astype(np.uint8)
        extracted_b64 = encode_bgr_to_png_b64(extracted_u8)

    return DetectResponse(
        facial_area=FacialArea(**face_info["facial_area"]),
        confidence=face_info.get("confidence"),
        detector_backend=state.detector_backend,
        extracted_png_base64=extracted_b64,
    )
