"""FastAPI 엔드포인트 통합 테스트.
무거운 DeepFace 호출은 모킹하고 라우팅/스키마/상태 흐름만 검증.
"""
from __future__ import annotations

import os
from io import BytesIO
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# lifespan에서 모델 warmup 스킵
os.environ.setdefault("SKIP_WARMUP", "1")


@pytest.fixture
def png_bytes() -> bytes:
    img = (np.ones((200, 200, 3)) * 128).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_status(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in {"ok", "loading"}
        assert isinstance(data["model_loaded"], bool)


class TestOptions:
    def test_options_lists(self, client):
        r = client.get("/api/options")
        assert r.status_code == 200
        data = r.json()
        assert "VGG-Face" in data["models"]
        assert "cosine" in data["metrics"]
        assert "opencv" in data["detectors"]


class TestSettings:
    def test_get_settings(self, client):
        r = client.get("/api/settings")
        assert r.status_code == 200
        data = r.json()
        assert data["model_name"] == "VGG-Face"
        assert data["distance_metric"] == "cosine"

    def test_update_settings_partial(self, client):
        r = client.put("/api/settings", json={"detector_backend": "ssd", "threshold": 0.7, "top_k": 5})
        assert r.status_code == 200
        data = r.json()
        assert data["detector_backend"] == "ssd"
        assert data["threshold"] == 0.7
        assert data["top_k"] == 5

    def test_update_settings_invalid_detector(self, client):
        r = client.put("/api/settings", json={"detector_backend": "nonexistent"})
        assert r.status_code == 422

    def test_update_settings_out_of_range_threshold(self, client):
        r = client.put("/api/settings", json={"threshold": 1.5})
        assert r.status_code == 422


class TestDatabase:
    def test_get_status(self, client):
        r = client.get("/api/database")
        assert r.status_code == 200
        data = r.json()
        assert "persons" in data
        assert "person_count" in data
        assert "total_faces" in data

    def test_add_face_missing_identity(self, client, png_bytes):
        r = client.post(
            "/api/database/faces",
            files={"image": ("t.png", png_bytes, "image/png")},
            data={"identity": "   "},
        )
        assert r.status_code == 422


class TestRecognize:
    def test_recognize_invalid_threshold(self, client, png_bytes):
        r = client.post(
            "/api/recognize",
            files={"image": ("t.png", png_bytes, "image/png")},
            data={"threshold": "1.5", "top_k": "1"},
        )
        assert r.status_code == 422

    def test_recognize_empty_file(self, client):
        r = client.post(
            "/api/recognize",
            files={"image": ("t.png", b"", "image/png")},
        )
        assert r.status_code == 400

    def test_recognize_undecodable(self, client):
        r = client.post(
            "/api/recognize",
            files={"image": ("t.png", b"not-an-image", "image/png")},
        )
        assert r.status_code == 400

    def test_recognize_oversize(self, client):
        big = b"x" * (5 * 1024 * 1024 + 1)
        r = client.post(
            "/api/recognize",
            files={"image": ("t.png", big, "image/png")},
        )
        assert r.status_code == 413

    @patch("api.routers.recognition.visualize_recognition")
    def test_recognize_happy_path(self, mock_vis, client, png_bytes):
        # 인식 시스템과 시각화 모킹
        mock_vis.return_value = np.ones((200, 200, 3), dtype=np.uint8) * 255

        with patch.object(client.app.state.app_state.system, "recognize_all_faces") as mock_rec:
            mock_rec.return_value = [
                {
                    "facial_area": {"x": 10, "y": 10, "w": 50, "h": 50},
                    "matches": [("alice", 0.95)],
                }
            ]
            r = client.post(
                "/api/recognize",
                files={"image": ("t.png", png_bytes, "image/png")},
            )
        assert r.status_code == 200
        data = r.json()
        assert len(data["faces"]) == 1
        assert data["faces"][0]["matches"][0]["identity"] == "alice"
        assert data["faces"][0]["matches"][0]["score"] == 0.95
        assert data["annotated_png_base64"] is not None
        assert len(data["annotated_png_base64"]) > 100


class TestDetect:
    @patch("api.routers.detection.extract_face")
    @patch("api.routers.detection.detect_face")
    def test_detect_happy_path(self, mock_det, mock_ext, client, png_bytes):
        mock_det.return_value = {"facial_area": {"x": 0, "y": 0, "w": 100, "h": 100}, "confidence": 0.92}
        mock_ext.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        r = client.post("/api/detect", files={"image": ("t.png", png_bytes, "image/png")})
        assert r.status_code == 200
        data = r.json()
        assert data["facial_area"]["w"] == 100
        assert data["confidence"] == 0.92
        assert data["extracted_png_base64"] is not None

    @patch("api.routers.detection.detect_face")
    def test_detect_no_face(self, mock_det, client, png_bytes):
        mock_det.return_value = None
        r = client.post("/api/detect", files={"image": ("t.png", png_bytes, "image/png")})
        assert r.status_code == 404
