from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 경로/런타임 초기화 (config 임포트 전에 cwd를 backend/로 고정해 상대경로 해소)
_BACKEND_DIR = Path(__file__).resolve().parents[1]
os.chdir(_BACKEND_DIR)

from config import configure_runtime, DATABASE_FILE, get_logger  # noqa: E402

configure_runtime()

from api.deps import AppState  # noqa: E402
from api.routers import database, detection, recognition, settings  # noqa: E402
from api.schemas import HealthResponse  # noqa: E402
from face_representation import represent_face  # noqa: E402

logger = get_logger("api.main")

_SKIP_WARMUP = os.environ.get("SKIP_WARMUP", "0").lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    base = os.path.splitext(DATABASE_FILE)[0]
    if os.path.exists(base + ".npz") and os.path.exists(base + ".json"):
        state.system.load_database(DATABASE_FILE)
    elif os.path.exists(base + ".pkl"):
        state.system.load_database(base + ".pkl")

    if not _SKIP_WARMUP:
        logger.info("DeepFace 모델 사전 로딩 (warmup)...")
        try:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            _ = represent_face(dummy, state.system.model_name)
            state.model_loaded = True
            logger.info("DeepFace warmup 완료")
        except Exception as e:
            logger.warning(f"warmup 실패 (지연 로딩으로 전환): {e}")
            state.model_loaded = False
    else:
        logger.info("SKIP_WARMUP=1 — warmup 생략, 첫 요청에서 모델 로드")

    app.state.app_state = state
    yield


app = FastAPI(
    title="Face Identification API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(recognition.router, prefix="/api", tags=["recognition"])
app.include_router(detection.router, prefix="/api", tags=["detection"])
app.include_router(database.router, prefix="/api/database", tags=["database"])
app.include_router(settings.router, prefix="/api", tags=["settings"])


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state: AppState = app.state.app_state
    return HealthResponse(
        status="ok" if state.model_loaded else "loading",
        model_loaded=state.model_loaded,
    )


# 프로덕션: frontend/dist를 정적 서빙. dev 환경에서는 dist 없음 → /api/* 만 동작
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        """SPA 라우터(History API) 지원: 모든 비-API 경로를 index.html로"""
        if full_path.startswith("api/"):
            return FileResponse(_FRONTEND_DIST / "index.html", status_code=404)
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
else:
    logger.info(f"frontend/dist 없음 — API 전용 모드 ({_FRONTEND_DIST})")
