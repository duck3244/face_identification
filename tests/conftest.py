import sys
import os
import pytest
import numpy as np

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_embedding():
    """테스트용 임의 임���딩 벡터 (VGG-Face 차원: 4096)"""
    return np.random.rand(4096).astype('float32').tolist()


@pytest.fixture
def sample_embeddings():
    """테스트용 여러 임베딩 벡터"""
    return [np.random.rand(4096).astype('float32').tolist() for _ in range(5)]


@pytest.fixture
def tmp_image(tmp_path):
    """테스트용 더미 이미지 파일 생성"""
    img_path = tmp_path / "test.jpg"
    try:
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
    except ImportError:
        # cv2가 없으면 빈 바이트로 더미 파일 생성
        img_path.write_bytes(b'\xff\xd8\xff\xe0')  # JPEG magic bytes
    return str(img_path)
