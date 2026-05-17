from __future__ import annotations

from dataclasses import dataclass, field

from config import DEFAULT_DETECTOR_BACKEND, DEFAULT_THRESHOLD, DEFAULT_TOP_K
from face_recognition import FaceRecognitionSystem


@dataclass
class AppState:
    """FastAPI 앱 수명 동안 유지되는 단일 사용자 상태"""
    system: FaceRecognitionSystem = field(default_factory=FaceRecognitionSystem)
    detector_backend: str = DEFAULT_DETECTOR_BACKEND
    default_threshold: float = DEFAULT_THRESHOLD
    default_top_k: int = DEFAULT_TOP_K
    model_loaded: bool = False
