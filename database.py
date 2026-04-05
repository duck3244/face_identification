from __future__ import annotations

import json
import os

import faiss
import numpy as np

from face_representation import represent_face
from validators import validate_threshold, validate_top_k
from config import get_logger

logger = get_logger(__name__)


class FaceDatabase:
    def __init__(self, model_name: str, distance_metric: str) -> None:
        """
        얼굴 데이터베이스 초기화

        Parameters:
        -----------
        model_name: str
            얼굴 표현을 추출하는 데 사용할 모델 이름
        distance_metric: str
            유사성 측정에 사용할 거리 측정 방법 (cosine, euclidean, euclidean_l2)
        """
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.embeddings: list[list[float]] = []
        self.identities: list[str] = []
        self.index: faiss.Index | None = None
        self._index_dirty: bool = False


    def add_face(self, img_path: str, identity: str) -> bool:
        """
        얼굴을 데이터베이스에 추가

        Parameters:
        -----------
        img_path: str
            데이터베이스에 추가할 얼굴 이미지 경로
        identity: str
            얼굴 이미지에 연결할 신원 정보

        Returns:
        --------
        bool
            성공 여부
        """
        embedding = represent_face(img_path, self.model_name)

        if embedding is not None:
            self.embeddings.append(embedding)
            self.identities.append(identity)
            self._index_dirty = True
            return True
        else:
            return False


    def build_index(self) -> None:
        """공개 인덱스 빌드 메서드"""
        self._build_index()
        self._index_dirty = False


    def _build_index(self) -> None:
        """FAISS 인덱스 구축"""
        if len(self.embeddings) == 0:
            logger.warning("데이터베이스가 비어 있습니다.")
            return

        embeddings_array = np.array(self.embeddings).astype('float32')
        dimension = embeddings_array.shape[1]

        if self.distance_metric == "cosine":
            faiss.normalize_L2(embeddings_array)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings_array)
        logger.info(f"FAISS 인덱스가 {len(self.embeddings)}개의 얼굴로 구축되었습니다.")


    def search(
        self,
        query_embedding: list[float] | np.ndarray,
        threshold: float,
        top_k: int
    ) -> list[tuple[str, float]]:
        """
        데이터베이스에서 유사한 얼굴 검색

        Parameters:
        -----------
        query_embedding: list[float] | np.ndarray
            검색할 쿼리 임베딩
        threshold: float
            얼굴 일치로 간주할 임계값
        top_k: int
            반환할 상위 일치 수

        Returns:
        --------
        list[tuple[str, float]]
            [(identity, similarity_score), ...] 형식의 일치 목록
        """
        validate_threshold(threshold)
        validate_top_k(top_k)

        if self._index_dirty:
            self.build_index()

        if self.index is None or len(self.embeddings) == 0:
            logger.warning("데이터베이스가 비어 있습니다.")
            return []

        query_array = np.array([query_embedding]).astype('float32')

        if self.distance_metric == "cosine":
            faiss.normalize_L2(query_array)

        distances, indices = self.index.search(query_array, top_k)

        results: list[tuple[str, float]] = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = distances[0][i]

            if self.distance_metric == "cosine":
                if score >= threshold:
                    results.append((self.identities[idx], float(score)))
            else:
                similarity = 1 / (1 + score)
                if similarity >= threshold:
                    results.append((self.identities[idx], float(similarity)))

        return results


    def save(self, file_path: str) -> None:
        """데이터베이스를 파일에 저장 (npz + json 형식)"""
        base_path = os.path.splitext(file_path)[0]
        npz_path = base_path + '.npz'
        json_path = base_path + '.json'

        embeddings_array = np.array(self.embeddings).astype('float32')
        np.savez_compressed(npz_path, embeddings=embeddings_array)

        metadata = {
            'identities': self.identities,
            'model_name': self.model_name,
            'distance_metric': self.distance_metric
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"데이터베이스가 {base_path}에 저장되었습니다.")


    def load(self, file_path: str) -> bool:
        """파일에서 데이터베이스 로드 (npz + json 또는 레거시 pkl 지원)"""
        base_path = os.path.splitext(file_path)[0]
        npz_path = base_path + '.npz'
        json_path = base_path + '.json'

        try:
            if os.path.exists(npz_path) and os.path.exists(json_path):
                return self._load_npz_json(npz_path, json_path)
            elif os.path.exists(file_path) and file_path.endswith('.pkl'):
                logger.info("레거시 pkl 형식을 감지했습니다. 새 형식으로 마이그레이션합니다...")
                return self._load_and_migrate_pkl(file_path)
            else:
                logger.error(f"데이터베이스 파일을 찾을 수 없습니다: {file_path}")
                return False
        except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"데이터베이스 로드 중 오류 발생: {e}")
            return False


    def _load_npz_json(self, npz_path: str, json_path: str) -> bool:
        """npz + json 형식에서 로드"""
        data = np.load(npz_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        if metadata['model_name'] != self.model_name:
            raise ValueError(
                f"모델 불일치: 데이터베이스는 '{metadata['model_name']}'을 사용하지만, "
                f"현재 시스템은 '{self.model_name}'을 사용합니다."
            )

        self.embeddings = data['embeddings'].tolist()
        self.identities = metadata['identities']
        self.model_name = metadata['model_name']
        self.distance_metric = metadata['distance_metric']

        self._build_index()
        self._index_dirty = False
        logger.info("데이터베이스가 로드되었습니다.")
        return True


    def _load_and_migrate_pkl(self, pkl_path: str) -> bool:
        """레거시 pkl 파일을 로드하고 새 형식으로 마이그레이션"""
        import pickle
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.embeddings = data['embeddings']
        self.identities = data['identities']
        self.model_name = data['model_name']
        self.distance_metric = data['distance_metric']

        self._build_index()
        self._index_dirty = False

        # 새 형식으로 저장
        self.save(pkl_path)
        logger.info("레거시 pkl 파일이 새 형식으로 마이그레이션되었습니다.")
        return True
