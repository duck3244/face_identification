from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.deps import AppState
from api.schemas import Options, Settings, SettingsUpdate
from config import DATABASE_FILE
from face_recognition import FaceRecognitionSystem

router = APIRouter()

MODELS = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
]
METRICS = ["cosine", "euclidean", "euclidean_l2"]
DETECTORS = [
    "opencv", "ssd", "dlib", "mtcnn", "retinaface",
    "mediapipe", "yolov8", "yunet", "centerface",
]


def _state(request: Request) -> AppState:
    return request.app.state.app_state


def _current(state: AppState) -> Settings:
    return Settings(
        model_name=state.system.model_name,
        distance_metric=state.system.distance_metric,
        detector_backend=state.detector_backend,
        threshold=state.default_threshold,
        top_k=state.default_top_k,
    )


@router.get("/options", response_model=Options)
def get_options() -> Options:
    return Options(models=MODELS, metrics=METRICS, detectors=DETECTORS)


@router.get("/settings", response_model=Settings)
def get_settings(request: Request) -> Settings:
    return _current(_state(request))


@router.put("/settings", response_model=Settings)
def update_settings(request: Request, payload: SettingsUpdate) -> Settings:
    state = _state(request)

    if payload.model_name is not None and payload.model_name not in MODELS:
        raise HTTPException(status_code=422, detail=f"지원하지 않는 모델: {payload.model_name}")
    if payload.detector_backend is not None and payload.detector_backend not in DETECTORS:
        raise HTTPException(status_code=422, detail=f"지원하지 않는 검출 백엔드: {payload.detector_backend}")

    if payload.detector_backend is not None:
        state.detector_backend = payload.detector_backend
    if payload.threshold is not None:
        state.default_threshold = payload.threshold
    if payload.top_k is not None:
        state.default_top_k = payload.top_k

    new_model = payload.model_name or state.system.model_name
    new_metric = payload.distance_metric or state.system.distance_metric
    if new_model != state.system.model_name or new_metric != state.system.distance_metric:
        state.system = FaceRecognitionSystem(new_model, new_metric)
        # 모델/메트릭 변경 시 기존 DB 로드 시도 (호환되지 않으면 빈 상태)
        import os
        base = os.path.splitext(DATABASE_FILE)[0]
        if os.path.exists(base + ".npz") and os.path.exists(base + ".json"):
            state.system.load_database(DATABASE_FILE)

    return _current(state)
