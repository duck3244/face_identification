import os
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name):
    return logging.getLogger(name)


# GPU 설정 (환경변수로 제어 가능)
USE_GPU = os.environ.get("FACE_USE_GPU", "false").lower() == "true"
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
if not USE_GPU:
    tf.config.set_visible_devices([], 'GPU')

# 기본 설정값 (환경변수로 오버라이드 가능)
DEFAULT_MODEL_NAME = os.environ.get("FACE_MODEL", "VGG-Face")
DEFAULT_DISTANCE_METRIC = os.environ.get("FACE_DISTANCE_METRIC", "cosine")
DEFAULT_DETECTOR_BACKEND = os.environ.get("FACE_DETECTOR", "opencv")
DEFAULT_THRESHOLD = float(os.environ.get("FACE_THRESHOLD", "0.5"))
DEFAULT_TOP_K = int(os.environ.get("FACE_TOP_K", "3"))
DEFAULT_TARGET_SIZE = (224, 224)

# 파일 경로 관련 설정
DATABASE_DIR = "face_database"
DATABASE_FILE = "face_database.pkl"
TEST_IMAGE_PATH = "test_image.jpg"
