import os

from config import (
    DEFAULT_MODEL_NAME, DEFAULT_DISTANCE_METRIC,
    DATABASE_DIR, DATABASE_FILE, get_logger,
)
from face_recognition import FaceRecognitionSystem

logger = get_logger(__name__)


def build_database():
    """얼굴 데이터베이스 구축"""
    logger.info("CPU 모드로 얼굴 인식 시스템을 초기화합니다...")
    face_system = FaceRecognitionSystem(model_name=DEFAULT_MODEL_NAME, distance_metric=DEFAULT_DISTANCE_METRIC)

    face_count = 0
    if os.path.exists(DATABASE_DIR):
        logger.info(f"'{DATABASE_DIR}' 디렉토리에서 얼굴 데이터베이스를 구축합니다...")
        person_count = 0

        for person_name in sorted(os.listdir(DATABASE_DIR)):
            person_dir = os.path.join(DATABASE_DIR, person_name)
            if os.path.isdir(person_dir):
                person_count += 1
                success_count = 0
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_file)
                        logger.info(f"{person_name}의 얼굴 등록 중: {img_file}")
                        if face_system.add_face_to_database(img_path, person_name):
                            face_count += 1
                            success_count += 1
                        else:
                            logger.warning(f"얼굴 등록 실패 (검출 불가 또는 임베딩 추출 실패): {img_path}")

                logger.info(f"{person_name}의 얼굴 {success_count}개가 성공적으로 등록되었습니다.")

        # 모든 얼굴 추가 후 인덱스 한 번만 빌드
        face_system.database.build_index()
        logger.info(f"총 {person_count}��의 인물에 대해 {face_count}개의 얼굴이 등록되었습니다.")
    else:
        logger.warning(f"'{DATABASE_DIR}' 디렉토리가 존재하지 않습니다. 디렉토리를 생성하고 얼굴 이미지를 추가하세요.")

    # 데이터베이스 저장
    if face_count > 0:
        face_system.save_database(DATABASE_FILE)
        logger.info("데이터베이스가 성공적으로 저장되었습니다.")
    else:
        logger.warning("등록된 얼굴이 없어 데이터베이스를 저장하지 않았습니다.")

    return face_system


if __name__ == "__main__":
    from config import configure_runtime
    configure_runtime()
    try:
        build_database()
    except (FileNotFoundError, RuntimeError, OSError) as e:
        logger.error(f"데이터베이스 구축 중 오류가 발생했습니다: {e}")
