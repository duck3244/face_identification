import os
import argparse

from config import DATABASE_FILE, TEST_IMAGE_PATH, get_logger, configure_runtime

configure_runtime()

from build_database import build_database
from face_recognition import FaceRecognitionSystem

logger = get_logger(__name__)


def _database_exists(db_file):
    """데이터베이스 파일 존재 여부 확인 (npz+json 또는 레거시 pkl)"""
    base_path = os.path.splitext(db_file)[0]
    npz_exists = os.path.exists(base_path + '.npz') and os.path.exists(base_path + '.json')
    pkl_exists = os.path.exists(base_path + '.pkl')
    return npz_exists or pkl_exists


def parse_args():
    parser = argparse.ArgumentParser(description="얼굴 인식 시스템")
    parser.add_argument("--model", default=None, help="얼굴 인식 모델 (VGG-Face, Facenet, ArcFace 등)")
    parser.add_argument("--threshold", type=float, default=0.6, help="얼굴 일치 임계값 (0.0~1.0)")
    parser.add_argument("--top-k", type=int, default=3, help="반환할 상위 일치 수")
    parser.add_argument("--image", default=None, help="테스트 이미지 경로")
    parser.add_argument("--database", default=None, help="데이터베이스 파일 경로")
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()
    db_file = args.database or DATABASE_FILE
    test_image = args.image or TEST_IMAGE_PATH

    try:
        logger.info("얼굴 인식 시스템을 초기화합니다...")
        if args.model:
            face_system = FaceRecognitionSystem(model_name=args.model)
        else:
            face_system = FaceRecognitionSystem()

        # 기존 데이터베이스가 있다면 로드, 없으면 새로 구축
        if _database_exists(db_file):
            logger.info("기존 데이터베이스를 로드합니다...")
            if not face_system.load_database(db_file):
                logger.warning("데이터베이스 로드 실패. 새로운 데이터베이스를 구축합니다...")
                face_system = build_database()
        else:
            logger.info("데이터베이스가 존재하지 않습니다. 새로운 데이터베이스를 구축합니다...")
            face_system = build_database()

        # 테스트 이미지로 얼굴 인식
        if os.path.exists(test_image):
            logger.info(f"테스트 이미지 '{test_image}'를 처리합니다...")

            # 얼굴 검출
            detected_face = face_system.detect_face(test_image)
            if detected_face is not None:
                logger.info("얼굴이 성공적으로 검출되었습니다.")

            # 얼굴 표현 추출
            face_embedding = face_system.represent_face(test_image)
            if face_embedding is not None:
                logger.debug(f"얼굴 임베딩 생성 성공! 차원: {len(face_embedding)}")

            # 얼굴 인식
            recognition_results = face_system.recognize_face(
                test_image, threshold=args.threshold, top_k=args.top_k
            )
            if recognition_results:
                logger.info("인식 결과:")
                for identity, score in recognition_results:
                    logger.info(f"  - {identity}: {score:.4f}")
            else:
                logger.info("일치하는 얼굴을 찾을 수 없습니다.")

            # 결과 시각화
            face_system.visualize_recognition(test_image, threshold=args.threshold, top_k=args.top_k)
        else:
            logger.warning(f"테스트 이미지 '{test_image}'가 존재하지 않습니다.")

    except Exception as e:
        logger.exception(f"실행 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
