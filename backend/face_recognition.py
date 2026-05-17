from __future__ import annotations

import numpy as np

from database import FaceDatabase
from face_representation import represent_face, represent_faces
from visualization import visualize_recognition
from face_detection import detect_face, extract_face
from config import DEFAULT_MODEL_NAME, DEFAULT_DISTANCE_METRIC, DEFAULT_THRESHOLD, DEFAULT_TOP_K, get_logger

logger = get_logger(__name__)


class FaceRecognitionSystem:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, distance_metric: str = DEFAULT_DISTANCE_METRIC) -> None:
        """
        얼굴 인식 시스템 초기화

        Parameters:
        -----------
        model_name: str
            얼굴 표현을 추출하는 데 사용할 모델 (VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace)
        distance_metric: str
            유사성 측정에 사용할 거리 측정 방법 (cosine, euclidean, euclidean_l2)
        """
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.database = FaceDatabase(model_name, distance_metric)


    def detect_face(self, img_path: str) -> dict | None:
        """이미지에서 얼굴 검출"""
        return detect_face(img_path)


    def extract_face(self, img_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray | None:
        """이미지에서 얼굴 영역만 추출"""
        return extract_face(img_path, target_size)


    def represent_face(self, img_path: str) -> list[float] | None:
        """이미지에서 얼굴 표현(임베딩) 추출"""
        return represent_face(img_path, self.model_name)


    def add_face_to_database(self, img_path: str, identity: str) -> bool:
        """얼굴을 데이터베이스에 추가"""
        return self.database.add_face(img_path, identity)


    def recognize_face(
        self,
        img_path: str,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K
    ) -> list[tuple[str, float]]:
        """
        이미지에서 얼굴을 인식하고 데이터베이스에서 일치하는 얼굴 찾기

        Parameters:
        -----------
        img_path: str
            인식할 얼굴 이미지 경로
        threshold: float
            얼굴 일치로 간주할 임계값 (코사인 거리의 경우 높을수록 더 유사함)
        top_k: int
            반환할 상위 일치 수

        Returns:
        --------
        list[tuple[str, float]]
            [(identity, similarity_score), ...] 형식의 일치 목록
        """
        query_embedding = self.represent_face(img_path)

        if query_embedding is None:
            logger.warning("쿼리 이미지에서 얼굴을 추출할 수 없습니다.")
            return []

        return self.database.search(query_embedding, threshold, top_k)


    def recognize_all_faces(
        self,
        img_path: str,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K
    ) -> list[dict]:
        """
        이미지의 모든 얼굴을 각각 인식

        Returns:
        --------
        list[dict]
            [{'facial_area': {x,y,w,h}, 'matches': [(identity, score), ...]}, ...]
        """
        face_objs = represent_faces(img_path, self.model_name)
        if not face_objs:
            return []
        return [
            {
                "facial_area": obj["facial_area"],
                "matches": self.database.search(obj["embedding"], threshold, top_k),
            }
            for obj in face_objs
        ]


    def save_database(self, file_path: str) -> None:
        """데이터베이스를 파일에 저장"""
        self.database.save(file_path)


    def load_database(self, file_path: str) -> bool:
        """파일에서 데이터베이스 로드"""
        return self.database.load(file_path)


    def visualize_recognition(
        self,
        img_path: str,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K
    ) -> np.ndarray | None:
        """얼굴 인식 결과 시각화 (모든 얼굴 box-label 매핑)"""
        face_results = self.recognize_all_faces(img_path, threshold, top_k)
        return visualize_recognition(img_path, face_results)
