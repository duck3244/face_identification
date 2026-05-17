from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Match(BaseModel):
    identity: str
    score: float


class FaceResult(BaseModel):
    facial_area: FacialArea
    matches: list[Match]


class RecognizeResponse(BaseModel):
    faces: list[FaceResult]
    annotated_png_base64: str | None = None


class DetectResponse(BaseModel):
    facial_area: FacialArea
    confidence: float | None
    detector_backend: str
    extracted_png_base64: str | None = None


class PersonStat(BaseModel):
    name: str
    face_count: int


class DatabaseStatus(BaseModel):
    persons: list[PersonStat]
    person_count: int
    total_faces: int


class AddFaceResponse(BaseModel):
    success: bool
    message: str
    db_status: DatabaseStatus


class RebuildResponse(BaseModel):
    success: bool
    person_count: int
    face_count: int
    db_status: DatabaseStatus


class Settings(BaseModel):
    model_name: str
    distance_metric: Literal["cosine", "euclidean", "euclidean_l2"]
    detector_backend: str
    threshold: float = Field(ge=0.0, le=1.0)
    top_k: int = Field(ge=1)


class SettingsUpdate(BaseModel):
    model_name: str | None = None
    distance_metric: Literal["cosine", "euclidean", "euclidean_l2"] | None = None
    detector_backend: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)


class Options(BaseModel):
    models: list[str]
    metrics: list[str]
    detectors: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    model_loaded: bool
