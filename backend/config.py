import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name):
    return logging.getLogger(name)


USE_GPU = os.environ.get("FACE_USE_GPU", "false").lower() == "true"

DEFAULT_MODEL_NAME = os.environ.get("FACE_MODEL", "VGG-Face")
DEFAULT_DISTANCE_METRIC = os.environ.get("FACE_DISTANCE_METRIC", "cosine")
DEFAULT_DETECTOR_BACKEND = os.environ.get("FACE_DETECTOR", "opencv")
DEFAULT_THRESHOLD = float(os.environ.get("FACE_THRESHOLD", "0.5"))
DEFAULT_TOP_K = int(os.environ.get("FACE_TOP_K", "3"))
DEFAULT_TARGET_SIZE = (224, 224)

DATABASE_DIR = "face_database"
DATABASE_FILE = "face_database.npz"
TEST_IMAGE_PATH = "test_image.jpg"

_runtime_configured = False


def configure_runtime() -> None:
    """GPU/TensorFlow 런타임 설정. entry point에서 명시적으로 호출."""
    global _runtime_configured
    if _runtime_configured:
        return
    _runtime_configured = True

    if not USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    import tensorflow as tf
    if not USE_GPU:
        tf.config.set_visible_devices([], 'GPU')
