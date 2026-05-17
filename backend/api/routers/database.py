from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.deps import AppState
from api.schemas import AddFaceResponse, DatabaseStatus, RebuildResponse
from api.utils import db_status_from_system, read_upload_as_bgr
from build_database import build_database
from config import DATABASE_FILE

router = APIRouter()


def _state(request: Request) -> AppState:
    return request.app.state.app_state


@router.get("", response_model=DatabaseStatus)
def get_status(request: Request) -> DatabaseStatus:
    return db_status_from_system(_state(request).system)


@router.post("/faces", response_model=AddFaceResponse)
def add_face(
    request: Request,
    image: UploadFile = File(...),
    identity: str = Form(...),
) -> AddFaceResponse:
    state = _state(request)
    name = identity.strip()
    if not name:
        raise HTTPException(status_code=422, detail="identity는 빈 문자열일 수 없습니다.")

    bgr = read_upload_as_bgr(image)
    success = state.system.add_face_to_database(bgr, name)
    if not success:
        return AddFaceResponse(
            success=False,
            message="얼굴 추가 실패. 이미지에서 얼굴을 검출할 수 없습니다.",
            db_status=db_status_from_system(state.system),
        )

    state.system.database.build_index()
    state.system.save_database(DATABASE_FILE)
    return AddFaceResponse(
        success=True,
        message=f"'{name}'의 얼굴이 추가되었습니다.",
        db_status=db_status_from_system(state.system),
    )


@router.post("/rebuild", response_model=RebuildResponse)
def rebuild(request: Request) -> RebuildResponse:
    state = _state(request)
    new_system = build_database()
    state.system = new_system
    status = db_status_from_system(new_system)
    return RebuildResponse(
        success=status.total_faces > 0,
        person_count=status.person_count,
        face_count=status.total_faces,
        db_status=status,
    )
