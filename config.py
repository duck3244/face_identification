import os
import tensorflow as tf


# GPU 메모리 제한 및 CPU 사용 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 비활성화
tf.config.set_visible_devices([], 'GPU')  # TensorFlow에서 GPU 사용 안함

# 기본 설정값
DEFAULT_MODEL_NAME = "VGG-Face"
DEFAULT_DISTANCE_METRIC = "cosine"
DEFAULT_DETECTOR_BACKEND = "opencv"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 1
DEFAULT_TARGET_SIZE = (224, 224)

# 파일 경로 관련 설정
DATABASE_DIR = "face_database"
DATABASE_FILE = "face_database.pkl"
TEST_IMAGE_PATH = "test_image.jpg"

