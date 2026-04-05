import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from face_detection import detect_face, extract_face


class TestDetectFace:
    @patch('face_detection.DeepFace')
    def test_detect_face_success(self, mock_deepface, tmp_image):
        mock_deepface.extract_faces.return_value = [
            {'face': np.zeros((224, 224, 3)), 'facial_area': {'x': 0, 'y': 0, 'w': 100, 'h': 100}}
        ]
        result = detect_face(tmp_image)
        assert result is not None
        assert 'facial_area' in result

    @patch('face_detection.DeepFace')
    def test_detect_face_no_face(self, mock_deepface, tmp_image):
        mock_deepface.extract_faces.return_value = []
        result = detect_face(tmp_image)
        assert result is None

    def test_detect_face_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            detect_face("/nonexistent/image.jpg")

    def test_detect_face_wrong_type(self):
        with pytest.raises(TypeError):
            detect_face(123)


class TestExtractFace:
    @patch('face_detection.DeepFace')
    def test_extract_face_success(self, mock_deepface, tmp_image):
        expected_face = np.zeros((224, 224, 3))
        mock_deepface.extract_faces.return_value = [{'face': expected_face}]
        result = extract_face(tmp_image)
        assert result is not None
        np.testing.assert_array_equal(result, expected_face)

    @patch('face_detection.DeepFace')
    def test_extract_face_no_face(self, mock_deepface, tmp_image):
        mock_deepface.extract_faces.return_value = []
        result = extract_face(tmp_image)
        assert result is None

    def test_extract_face_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            extract_face("/nonexistent/image.jpg")
