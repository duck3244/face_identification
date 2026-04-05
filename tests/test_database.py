import json
import os
import pytest
import numpy as np
from unittest.mock import patch

from database import FaceDatabase


class TestFaceDatabase:
    def test_init(self):
        db = FaceDatabase("VGG-Face", "cosine")
        assert db.model_name == "VGG-Face"
        assert db.distance_metric == "cosine"
        assert db.embeddings == []
        assert db.identities == []
        assert db.index is None

    def test_build_index_cosine(self, sample_embeddings):
        db = FaceDatabase("VGG-Face", "cosine")
        db.embeddings = sample_embeddings
        db.identities = [f"person_{i}" for i in range(len(sample_embeddings))]
        db.build_index()
        assert db.index is not None
        assert db.index.ntotal == len(sample_embeddings)

    def test_build_index_euclidean(self, sample_embeddings):
        db = FaceDatabase("VGG-Face", "euclidean")
        db.embeddings = sample_embeddings
        db.identities = [f"person_{i}" for i in range(len(sample_embeddings))]
        db.build_index()
        assert db.index is not None
        assert db.index.ntotal == len(sample_embeddings)

    def test_build_index_empty(self):
        db = FaceDatabase("VGG-Face", "cosine")
        db.build_index()
        assert db.index is None

    def test_search_empty_database(self, sample_embedding):
        db = FaceDatabase("VGG-Face", "cosine")
        results = db.search(sample_embedding, threshold=0.5, top_k=3)
        assert results == []

    def test_search_cosine(self, sample_embedding):
        db = FaceDatabase("VGG-Face", "cosine")
        db.embeddings = [sample_embedding]
        db.identities = ["test_person"]
        db.build_index()

        # 동일한 임베딩으로 검색하면 높은 유사도
        results = db.search(sample_embedding, threshold=0.5, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "test_person"
        assert results[0][1] >= 0.5

    def test_search_euclidean(self, sample_embedding):
        db = FaceDatabase("VGG-Face", "euclidean")
        db.embeddings = [sample_embedding]
        db.identities = ["test_person"]
        db.build_index()

        results = db.search(sample_embedding, threshold=0.5, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "test_person"

    def test_search_with_threshold_filter(self, sample_embeddings):
        db = FaceDatabase("VGG-Face", "cosine")
        db.embeddings = sample_embeddings
        db.identities = [f"person_{i}" for i in range(len(sample_embeddings))]
        db.build_index()

        # 매우 높은 임계값으로 검색
        results = db.search(sample_embeddings[0], threshold=0.99, top_k=5)
        # 자기 자신만 매칭되거나 아무것도 안 됨
        assert len(results) <= 1

    def test_search_invalid_threshold(self, sample_embedding):
        db = FaceDatabase("VGG-Face", "cosine")
        with pytest.raises(ValueError):
            db.search(sample_embedding, threshold=1.5, top_k=1)

    def test_search_invalid_top_k(self, sample_embedding):
        db = FaceDatabase("VGG-Face", "cosine")
        with pytest.raises(ValueError):
            db.search(sample_embedding, threshold=0.5, top_k=0)

    def test_lazy_index_build(self, sample_embedding):
        """add_face 후 search 시 자동으로 인덱스 빌드"""
        db = FaceDatabase("VGG-Face", "cosine")
        db.embeddings = [sample_embedding]
        db.identities = ["test_person"]
        db._index_dirty = True

        assert db.index is None
        results = db.search(sample_embedding, threshold=0.5, top_k=1)
        assert db.index is not None
        assert len(results) >= 1


class TestFaceDatabasePersistence:
    def test_save_load_roundtrip(self, tmp_path, sample_embeddings):
        """저장/로드 왕복 데이터 일관성"""
        db = FaceDatabase("VGG-Face", "cosine")
        db.embeddings = sample_embeddings
        db.identities = [f"person_{i}" for i in range(len(sample_embeddings))]
        db.build_index()

        file_path = str(tmp_path / "test_db.pkl")
        db.save(file_path)

        # npz + json 파일이 생성되었는지 확인
        base_path = str(tmp_path / "test_db")
        assert os.path.exists(base_path + '.npz')
        assert os.path.exists(base_path + '.json')

        # 새 데이터베이스에 로드
        db2 = FaceDatabase("VGG-Face", "cosine")
        assert db2.load(file_path)

        assert len(db2.embeddings) == len(db.embeddings)
        assert db2.identities == db.identities
        assert db2.model_name == db.model_name
        assert db2.distance_metric == db.distance_metric
        np.testing.assert_array_almost_equal(
            np.array(db2.embeddings), np.array(db.embeddings), decimal=5
        )

    def test_load_model_mismatch(self, tmp_path, sample_embeddings):
        """모델 불일치 시 에러"""
        db = FaceDatabase("VGG-Face", "cosine")
        db.embeddings = sample_embeddings
        db.identities = ["person_0"] * len(sample_embeddings)
        db.build_index()

        file_path = str(tmp_path / "test_db.pkl")
        db.save(file_path)

        # 다른 모델로 로드 시도
        db2 = FaceDatabase("Facenet", "cosine")
        assert db2.load(file_path) is False

    def test_load_nonexistent_file(self):
        db = FaceDatabase("VGG-Face", "cosine")
        assert db.load("/nonexistent/file.pkl") is False

    def test_save_creates_correct_json(self, tmp_path, sample_embeddings):
        db = FaceDatabase("VGG-Face", "cosine")
        db.embeddings = sample_embeddings
        db.identities = ["alice", "bob", "charlie", "dave", "eve"]

        file_path = str(tmp_path / "test_db.pkl")
        db.save(file_path)

        json_path = str(tmp_path / "test_db.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        assert metadata['model_name'] == "VGG-Face"
        assert metadata['distance_metric'] == "cosine"
        assert metadata['identities'] == ["alice", "bob", "charlie", "dave", "eve"]
