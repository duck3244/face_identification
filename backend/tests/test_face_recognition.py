import numpy as np
import pytest
from unittest.mock import patch

from face_recognition import FaceRecognitionSystem


class TestFaceRecognitionSystem:
    def test_init(self):
        system = FaceRecognitionSystem("VGG-Face", "cosine")
        assert system.model_name == "VGG-Face"
        assert system.distance_metric == "cosine"
        assert system.database.model_name == "VGG-Face"

    @patch('face_recognition.represent_face')
    def test_recognize_face_no_face(self, mock_represent, tmp_image):
        mock_represent.return_value = None
        system = FaceRecognitionSystem("VGG-Face", "cosine")
        assert system.recognize_face(tmp_image) == []

    @patch('face_recognition.represent_face')
    def test_recognize_face_returns_matches(self, mock_represent, tmp_image, sample_embedding):
        mock_represent.return_value = sample_embedding
        system = FaceRecognitionSystem("VGG-Face", "cosine")
        system.database.embeddings = [sample_embedding]
        system.database.identities = ["alice"]
        system.database.build_index()

        results = system.recognize_face(tmp_image, threshold=0.5, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "alice"

    @patch('face_recognition.represent_faces')
    def test_recognize_all_faces_empty(self, mock_represent, tmp_image):
        mock_represent.return_value = None
        system = FaceRecognitionSystem("VGG-Face", "cosine")
        assert system.recognize_all_faces(tmp_image) == []

    @patch('face_recognition.represent_faces')
    def test_recognize_all_faces_per_face_results(self, mock_represent, tmp_image, sample_embedding):
        # 두 얼굴: 첫 번째는 DB에 있는 얼굴, 두 번째는 무작위
        other_embedding = np.random.rand(4096).astype('float32').tolist()
        mock_represent.return_value = [
            {"embedding": sample_embedding, "facial_area": {"x": 0, "y": 0, "w": 50, "h": 50}},
            {"embedding": other_embedding, "facial_area": {"x": 100, "y": 0, "w": 50, "h": 50}},
        ]
        system = FaceRecognitionSystem("VGG-Face", "cosine")
        system.database.embeddings = [sample_embedding]
        system.database.identities = ["alice"]
        system.database.build_index()

        face_results = system.recognize_all_faces(tmp_image, threshold=0.99, top_k=1)
        assert len(face_results) == 2
        # 각 얼굴에 자기 facial_area가 매핑되어야 함
        assert face_results[0]["facial_area"]["x"] == 0
        assert face_results[1]["facial_area"]["x"] == 100
        # 첫 얼굴은 alice와 정확히 매칭, 두 번째는 임계값 미달
        assert face_results[0]["matches"][0][0] == "alice"
        assert face_results[1]["matches"] == []
