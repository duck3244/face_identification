import cv2
import numpy as np
import pytest

from visualization import visualize_recognition


@pytest.fixture
def real_image(tmp_path):
    """visualize_recognition은 cv2.imread를 호출하므로 실제 디코딩 가능한 JPEG 필요."""
    img_path = tmp_path / "real.jpg"
    img = (np.ones((200, 200, 3)) * 128).astype(np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)


class TestVisualizeRecognition:
    def test_no_faces_returns_unchanged_image(self, real_image):
        out = visualize_recognition(real_image, [])
        assert out is not None
        assert out.shape == (200, 200, 3)

    def test_draws_matched_face(self, real_image):
        face_results = [
            {"facial_area": {"x": 10, "y": 10, "w": 80, "h": 80}, "matches": [("alice", 0.95)]}
        ]
        out = visualize_recognition(real_image, face_results)
        assert out is not None
        # 박스 영역에 녹색 픽셀이 존재 (선 두께 2)
        green_present = np.any((out[:, :, 1] > 200) & (out[:, :, 0] < 50) & (out[:, :, 2] < 50))
        assert green_present

    def test_draws_unknown_face(self, real_image):
        face_results = [
            {"facial_area": {"x": 10, "y": 10, "w": 80, "h": 80}, "matches": []}
        ]
        out = visualize_recognition(real_image, face_results)
        assert out is not None
        # 매칭 없으면 빨간 박스
        red_present = np.any((out[:, :, 0] > 200) & (out[:, :, 1] < 50) & (out[:, :, 2] < 50))
        assert red_present

    def test_multiple_faces_independent(self, real_image):
        face_results = [
            {"facial_area": {"x": 10, "y": 10, "w": 40, "h": 40}, "matches": [("a", 0.9)]},
            {"facial_area": {"x": 100, "y": 100, "w": 40, "h": 40}, "matches": [("b", 0.8)]},
        ]
        out = visualize_recognition(real_image, face_results)
        assert out is not None

    def test_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            visualize_recognition("/nonexistent/image.jpg", [])
