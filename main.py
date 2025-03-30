import os

from build_database import build_database
from face_recognition import FaceRecognitionSystem
from config import DATABASE_FILE, TEST_IMAGE_PATH


def main():
    """메인 실행 함수"""
    try:
        print("CPU 모드로 얼굴 인식 시스템을 초기화합니다...")
        # 얼굴 인식 시스템 초기화
        face_system = FaceRecognitionSystem()

        # 기존 데이터베이스가 있다면 로드
        face_count = 0
        if os.path.exists(DATABASE_FILE):
            print("기존 데이터베이스를 로드합니다...")
            if face_system.load_database(DATABASE_FILE):
                print("데이터베이스 로드 성공!")
                face_count = len(face_system.database.embeddings)
            else:
                print("데이터베이스 로드 실패. 새로운 데이터베이스를 구축합니다...")
                face_system = build_database()
                face_count = len(face_system.database.embeddings)
        else:
            print("데이터베이스가 존재하지 않습니다. 새로운 데이터베이스를 구축합니다...")
            face_system = build_database()
            face_count = len(face_system.database.embeddings)

        # 테스트 이미지로 얼굴 인식
        if os.path.exists(TEST_IMAGE_PATH):
            print(f"테스트 이미지 '{TEST_IMAGE_PATH}'를 처리합니다...")

            # 얼굴 검출
            detected_face = face_system.detect_face(TEST_IMAGE_PATH)
            if detected_face is not None:
                print("얼굴이 성공적으로 검출되었습니다.")

            # 얼굴 표현 추출
            face_embedding = face_system.represent_face(TEST_IMAGE_PATH)
            if face_embedding is not None:
                print(f"얼굴 임베딩 생성 성공! 차원: {len(face_embedding)}")

            # 얼굴 인식
            recognition_results = face_system.recognize_face(TEST_IMAGE_PATH, threshold=0.6, top_k=3)
            if recognition_results:
                print("인식 결과:")
                for identity, score in recognition_results:
                    print(f"  - {identity}: {score:.4f}")
            else:
                print("일치하는 얼굴을 찾을 수 없습니다.")

            # 결과 시각화
            face_system.visualize_recognition(TEST_IMAGE_PATH, threshold=0.6, top_k=3)
        else:
            print(f"테스트 이미지 '{TEST_IMAGE_PATH}'가 존재하지 않습니다.")

    except Exception as e:
        print(f"실행 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()

