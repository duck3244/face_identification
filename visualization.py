import cv2
import matplotlib.pyplot as plt

from deepface import DeepFace
from config import DEFAULT_DETECTOR_BACKEND


def visualize_recognition(img_path, recognition_results, detector_backend=DEFAULT_DETECTOR_BACKEND):
    """
    얼굴 인식 결과 시각화

    Parameters:
    -----------
    img_path: str
        인식할 얼굴 이미지 경로
    recognition_results: list
        [(identity, similarity_score), ...] 형식의 인식 결과 목록
    detector_backend: str
        사용할 얼굴 검출 백엔드
    """
    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not recognition_results:
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title("인식된 얼굴 없음")
        plt.axis('off')
        plt.show()
        return

    try:
        # DeepFace를 사용하여 얼굴 감지 (CPU 모드)
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        plt.figure(figsize=(12, 8))
        plt.imshow(img)

        for i, face_obj in enumerate(faces):
            if i >= len(recognition_results):
                break

            face_data = face_obj
            x, y, w, h = face_data['facial_area']['x'], face_data['facial_area']['y'], \
                face_data['facial_area']['w'], face_data['facial_area']['h']

            identity, score = recognition_results[i]

            # 얼굴 영역에 사각형 그리기
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
            plt.gca().add_patch(rect)

            # 인식 결과 텍스트 표시
            plt.text(x, y - 10, f"{identity} ({score:.2f})",
                     color='white', fontsize=12, backgroundcolor='green')

        plt.title("얼굴 인식 결과")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
        # 오류 발생 시 기본 이미지만 표시
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title("얼굴 인식 결과 (얼굴 검출 오류)")
        plt.axis('off')
        plt.show()

