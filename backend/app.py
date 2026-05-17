from __future__ import annotations

import os
from config import (
    DEFAULT_MODEL_NAME, DEFAULT_DISTANCE_METRIC, DEFAULT_DETECTOR_BACKEND,
    DEFAULT_THRESHOLD, DEFAULT_TOP_K, DATABASE_FILE, DATABASE_DIR,
    get_logger, configure_runtime,
)

configure_runtime()

import cv2
import gradio as gr
import numpy as np
from collections import Counter

from face_recognition import FaceRecognitionSystem
from build_database import build_database
from visualization import visualize_recognition

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.system = FaceRecognitionSystem()
        self.detector_backend = DEFAULT_DETECTOR_BACKEND
        self.default_threshold = DEFAULT_THRESHOLD
        self.default_top_k = DEFAULT_TOP_K
        self._load_db()

    def _load_db(self):
        base = os.path.splitext(DATABASE_FILE)[0]
        if os.path.exists(base + '.npz') and os.path.exists(base + '.json'):
            self.system.load_database(DATABASE_FILE)
        elif os.path.exists(base + '.pkl'):
            self.system.load_database(base + '.pkl')

    def reinitialize(self, model_name: str, distance_metric: str):
        self.system = FaceRecognitionSystem(model_name, distance_metric)
        self._load_db()


state = AppState()

# ──────────────────────────────────────────────
# HTML 테이블 유틸
# ──────────────────────────────────────────────

def _make_html_table(headers: list[str], rows: list[list[str]]) -> str:
    style = (
        "width:100%; border-collapse:collapse; font-size:14px;"
    )
    th_style = (
        "padding:8px 12px; text-align:left; border-bottom:2px solid #ddd; "
        "background:#f7f7f7; font-weight:600;"
    )
    td_style = "padding:8px 12px; border-bottom:1px solid #eee;"
    html = f'<table style="{style}"><thead><tr>'
    for h in headers:
        html += f'<th style="{th_style}">{h}</th>'
    html += '</tr></thead><tbody>'
    if not rows:
        html += f'<tr><td style="{td_style}" colspan="{len(headers)}">데이터 없음</td></tr>'
    else:
        for row in rows:
            html += '<tr>'
            for cell in row:
                html += f'<td style="{td_style}">{cell}</td>'
            html += '</tr>'
    html += '</tbody></table>'
    return html

# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────

def _get_db_status() -> tuple[str, str]:
    identities = state.system.database.identities
    counts = Counter(identities)
    person_count = len(counts)
    face_count = len(identities)
    status = f"인물 수: {person_count}명 / 총 얼굴 수: {face_count}개"
    rows = [[name, str(count)] for name, count in sorted(counts.items())]
    table_html = _make_html_table(["인물", "얼굴 수"], rows)
    return status, table_html

# ──────────────────────────────────────────────
# Tab 1: 얼굴 인식
# ──────────────────────────────────────────────

def fn_recognize(img_path: str, threshold: float, top_k: int):
    if img_path is None:
        return None, ""
    top_k = int(top_k)
    face_results = state.system.recognize_all_faces(img_path, threshold, top_k)
    annotated = visualize_recognition(img_path, face_results)
    rows: list[list[str]] = []
    for face_idx, face in enumerate(face_results, start=1):
        if face["matches"]:
            for identity, score in face["matches"]:
                rows.append([str(face_idx), identity, f"{score:.4f}"])
        else:
            rows.append([str(face_idx), "일치 없음", "-"])
    if not rows:
        rows = [["-", "얼굴 미검출", "-"]]
    table_html = _make_html_table(["얼굴 #", "인물", "유사도"], rows)
    return annotated, table_html

# ──────────────────────────────────────────────
# Tab 2: 데이터베이스 관리
# ──────────────────────────────────────────────

def fn_db_status():
    return _get_db_status()


def fn_add_face(img_path: str, person_name: str):
    if img_path is None or not person_name or not person_name.strip():
        return "이미지와 인물 이름을 모두 입력해주세요.", *_get_db_status()
    name = person_name.strip()
    success = state.system.add_face_to_database(img_path, name)
    if success:
        state.system.database.build_index()
        state.system.save_database(DATABASE_FILE)
        return f"'{name}'의 얼굴이 추가되었습니다.", *_get_db_status()
    else:
        return "얼굴 추가 실패. 이미지에서 얼굴을 검출할 수 없습니다.", *_get_db_status()


def fn_rebuild_db():
    gr.Info("데이터베이스 재구축 중...")
    new_system = build_database()
    state.system = new_system
    return "데이터베이스가 재구축되었습니다.", *_get_db_status()

# ──────────────────────────────────────────────
# Tab 3: 얼굴 검출/추출
# ──────────────────────────────────────────────

def fn_detect(img_path: str):
    if img_path is None:
        return None, None, "이미지를 업로드해주세요."

    img = cv2.imread(img_path)
    if img is None:
        return None, None, "이미지를 로드할 수 없습니다."
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    from face_detection import detect_face as detect_face_fn, extract_face as extract_face_fn
    face_info = detect_face_fn(img_path, detector_backend=state.detector_backend)
    if face_info is None:
        return img_rgb, None, "얼굴이 검출되지 않았습니다."

    annotated = img_rgb.copy()
    area = face_info['facial_area']
    x, y, w, h = area['x'], area['y'], area['w'], area['h']
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    confidence = face_info.get('confidence', 'N/A')
    info = f"위치: x={x}, y={y}, w={w}, h={h}\n신뢰도: {confidence}\n검출 백엔드: {state.detector_backend}"

    extracted = extract_face_fn(img_path, detector_backend=state.detector_backend)
    extracted_img = None
    if extracted is not None:
        extracted_img = (extracted * 255).astype(np.uint8) if extracted.max() <= 1.0 else extracted.astype(np.uint8)

    return annotated, extracted_img, info

# ──────────────────────────────────────────────
# Tab 4: 설정
# ──────────────────────────────────────────────

def fn_apply_settings(model_name: str, distance_metric: str, detector_backend: str, threshold: float, top_k: int):
    top_k_int = int(top_k)
    state.detector_backend = detector_backend
    state.default_threshold = threshold
    state.default_top_k = top_k_int

    old_model = state.system.model_name
    if model_name != old_model or distance_metric != state.system.distance_metric:
        try:
            state.reinitialize(model_name, distance_metric)
            msg = f"설정 적용 완료. 모델: {model_name}, 메트릭: {distance_metric}, 백엔드: {detector_backend}"
            if len(state.system.database.embeddings) == 0:
                msg += "\n⚠ 모델이 변경되어 기존 DB를 로드할 수 없습니다. DB 재구축이 필요합니다."
        except Exception as e:
            msg = f"설정 적용 중 오류: {e}"
    else:
        msg = f"설정 적용 완료. 백엔드: {detector_backend}, 임계값: {threshold}, Top-K: {top_k_int}"

    return msg, gr.update(value=threshold), gr.update(value=top_k_int)

# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"]
METRICS = ["cosine", "euclidean", "euclidean_l2"]
DETECTORS = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8", "yunet", "centerface"]


def create_app() -> gr.Blocks:
    init_status, init_table_html = _get_db_status()

    with gr.Blocks(title="얼굴 인식 시스템", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 얼굴 인식 시스템\nDeepFace + FAISS 기반 얼굴 인식")

        with gr.Tabs():
            # ── Tab 1: 얼굴 인식 ──
            with gr.Tab("얼굴 인식"):
                with gr.Row():
                    with gr.Column(scale=1):
                        rec_image = gr.Image(type="filepath", label="이미지 업로드")
                        rec_threshold = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.05, label="임계값")
                        rec_topk = gr.Number(value=DEFAULT_TOP_K, label="Top-K", precision=0)
                        rec_btn = gr.Button("인식 실행", variant="primary")
                    with gr.Column(scale=1):
                        rec_result_img = gr.Image(label="인식 결과")
                        rec_result_table = gr.HTML(label="매칭 결과")
                rec_btn.click(
                    fn=fn_recognize,
                    inputs=[rec_image, rec_threshold, rec_topk],
                    outputs=[rec_result_img, rec_result_table]
                )

            # ── Tab 2: 데이터베이스 관리 ──
            with gr.Tab("데이터베이스 관리"):
                gr.Markdown("### 현재 데이터베이스 상태")
                db_status_text = gr.Textbox(value=init_status, label="상태", interactive=False)
                db_status_table = gr.HTML(value=init_table_html)
                db_refresh_btn = gr.Button("상태 새로고침")
                db_refresh_btn.click(fn=fn_db_status, outputs=[db_status_text, db_status_table])

                gr.Markdown("---")
                gr.Markdown("### 새 얼굴 추가")
                with gr.Row():
                    add_image = gr.Image(type="filepath", label="얼굴 이미지")
                    with gr.Column():
                        add_name = gr.Textbox(label="인물 이름")
                        add_btn = gr.Button("얼굴 추가", variant="primary")
                add_result = gr.Textbox(label="결과", interactive=False)
                add_btn.click(
                    fn=fn_add_face,
                    inputs=[add_image, add_name],
                    outputs=[add_result, db_status_text, db_status_table]
                )

                gr.Markdown("---")
                gr.Markdown("### 데이터베이스 재구축")
                rebuild_btn = gr.Button("face_database/ 디렉토리에서 재구축", variant="stop")
                rebuild_result = gr.Textbox(label="결과", interactive=False)
                rebuild_btn.click(
                    fn=fn_rebuild_db,
                    outputs=[rebuild_result, db_status_text, db_status_table]
                )

            # ── Tab 3: 얼굴 검출/추출 ──
            with gr.Tab("얼굴 검출/추출"):
                with gr.Row():
                    with gr.Column(scale=1):
                        det_image = gr.Image(type="filepath", label="이미지 업로드")
                        det_btn = gr.Button("얼굴 검출", variant="primary")
                    with gr.Column(scale=1):
                        det_result_img = gr.Image(label="검출 결과")
                        det_face_img = gr.Image(label="추출된 얼굴")
                        det_info = gr.Textbox(label="검출 정보", interactive=False)
                det_btn.click(
                    fn=fn_detect,
                    inputs=[det_image],
                    outputs=[det_result_img, det_face_img, det_info]
                )

            # ── Tab 4: 설정 ──
            with gr.Tab("설정"):
                gr.Markdown("### 시스템 설정")
                gr.Markdown("> 모델 변경 시 기존 데이터베이스와 호환되지 않을 수 있습니다. 변경 후 DB 재구축이 필요합니다.")
                with gr.Row():
                    set_model = gr.Dropdown(choices=MODELS, value=DEFAULT_MODEL_NAME, label="인식 모델")
                    set_metric = gr.Dropdown(choices=METRICS, value=DEFAULT_DISTANCE_METRIC, label="거리 메트릭")
                    set_detector = gr.Dropdown(choices=DETECTORS, value=DEFAULT_DETECTOR_BACKEND, label="검출 백엔드")
                with gr.Row():
                    set_threshold = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.05, label="기본 임계값")
                    set_topk = gr.Number(value=DEFAULT_TOP_K, label="기본 Top-K", precision=0)
                set_btn = gr.Button("설정 적용", variant="primary")
                set_result = gr.Textbox(label="상태", interactive=False)
                set_btn.click(
                    fn=fn_apply_settings,
                    inputs=[set_model, set_metric, set_detector, set_threshold, set_topk],
                    outputs=[set_result, rec_threshold, rec_topk]
                )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
