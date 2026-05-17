import pytest
from unittest.mock import patch

from face_representation import represent_face, represent_faces


class TestRepresentFaces:
    @patch('face_representation.DeepFace')
    def test_multiple_faces(self, mock_deepface, tmp_image):
        mock_deepface.represent.return_value = [
            {"embedding": [0.1] * 4096, "facial_area": {"x": 0, "y": 0, "w": 50, "h": 50}, "face_confidence": 0.9},
            {"embedding": [0.2] * 4096, "facial_area": {"x": 60, "y": 0, "w": 50, "h": 50}, "face_confidence": 0.8},
        ]
        result = represent_faces(tmp_image)
        assert result is not None
        assert len(result) == 2
        assert result[0]["facial_area"]["x"] == 0
        assert result[1]["facial_area"]["x"] == 60

    @patch('face_representation.DeepFace')
    def test_no_face(self, mock_deepface, tmp_image):
        mock_deepface.represent.return_value = []
        assert represent_faces(tmp_image) is None

    @patch('face_representation.DeepFace')
    def test_deepface_error(self, mock_deepface, tmp_image):
        mock_deepface.represent.side_effect = ValueError("model load fail")
        assert represent_faces(tmp_image) is None

    def test_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            represent_faces("/nonexistent/image.jpg")


class TestRepresentFace:
    @patch('face_representation.DeepFace')
    def test_returns_first_embedding(self, mock_deepface, tmp_image):
        mock_deepface.represent.return_value = [
            {"embedding": [0.1] * 4096, "facial_area": {"x": 0, "y": 0, "w": 50, "h": 50}},
            {"embedding": [0.2] * 4096, "facial_area": {"x": 60, "y": 0, "w": 50, "h": 50}},
        ]
        result = represent_face(tmp_image)
        assert result is not None
        assert len(result) == 4096
        assert result[0] == pytest.approx(0.1)

    @patch('face_representation.DeepFace')
    def test_no_face_returns_none(self, mock_deepface, tmp_image):
        mock_deepface.represent.return_value = []
        assert represent_face(tmp_image) is None
