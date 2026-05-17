from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.deps import AppState
from api.schemas import FaceResult, FacialArea, Match, RecognizeResponse
from api.utils import encode_rgb_to_png_b64, read_upload_as_bgr
from visualization import visualize_recognition

router = APIRouter()


def _state(request: Request) -> AppState:
    return request.app.state.app_state


@router.post("/recognize", response_model=RecognizeResponse)
def recognize(
    request: Request,
    image: UploadFile = File(...),
    threshold: float | None = Form(default=None),
    top_k: int | None = Form(default=None),
) -> RecognizeResponse:
    state = _state(request)
    bgr = read_upload_as_bgr(image)

    th = threshold if threshold is not None else state.default_threshold
    k = top_k if top_k is not None else state.default_top_k

    if not (0.0 <= th <= 1.0):
        raise HTTPException(status_code=422, detail="threshold는 0.0~1.0 범위여야 합니다.")
    if k < 1:
        raise HTTPException(status_code=422, detail="top_k는 1 이상이어야 합니다.")

    face_results = state.system.recognize_all_faces(bgr, th, k)
    annotated_rgb = visualize_recognition(bgr, face_results)

    return RecognizeResponse(
        faces=[
            FaceResult(
                facial_area=FacialArea(**fr["facial_area"]),
                matches=[Match(identity=i, score=s) for i, s in fr["matches"]],
            )
            for fr in face_results
        ],
        annotated_png_base64=encode_rgb_to_png_b64(annotated_rgb) if annotated_rgb is not None else None,
    )
