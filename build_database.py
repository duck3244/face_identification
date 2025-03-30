import os

from face_recognition import FaceRecognitionSystem
from config import DEFAULT_MODEL_NAME, DEFAULT_DISTANCE_METRIC, DATABASE_DIR, DATABASE_FILE


def build_database():
    """얼굴 데이터베이스 구축"""
    print("CPU 모드로 얼굴 인식 시스템을 초기화합니다...")
    face_system = FaceRecognitionSystem(model_name=DEFAULT_MODEL_NAME, distance_metric=DEFAULT_DISTANCE_METRIC)

    if os.path.exists(DATABASE_DIR):
        print(f"'{DATABASE_DIR}' 디렉토리에서 얼굴 데이터베이스를 구축합니다...")
        person_count = 0
        face_count = 0

        for person_name in os.listdir(DATABASE_DIR):
            person_dir = os.path.join(DATABASE_DIR, person_name)
            if os.path.isdir(person_dir):
                person_count += 1
                success_count = 0
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_file)
                        print(f"{person_name}의 얼굴 등록 중: {img_file}")
                        if face_system.add_face_to_database(img_path, person_name):
                            face_count += 1
                            success_count += 1

                print(f"{person_name}의 얼굴 {success_count}개가 성공적으로 등록되었습니다.")

        print(f"총 {person_count}명의 인물에 대해 {face_count}개의 얼굴이 등록되었습니다.")
    else:
        print(f"'{DATABASE_DIR}' 디렉토리가 존재하지 않습니다. 디렉토리를 생성하고 얼굴 이미지를 추가하세요.")

    # 데이터베이스 저장
    if face_count > 0:
        face_system.save_database(DATABASE_FILE)
        print("데이터베이스가 성공적으로 저장되었습니다.")
    else:
        print("등록된 얼굴이 없어 데이터베이스를 저장하지 않았습니다.")

    return face_system


if __name__ == "__main__":
    try:
        build_database()
    except Exception as e:
        print(f"데이터베이스 구축 중 오류가 발생했습니다: {e}")

