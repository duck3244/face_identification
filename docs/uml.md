# UML 다이어그램

> Face Identification — 클래스/시퀀스/상태/컴포넌트/유스케이스 UML
> Mermaid 기반 (GitHub, VSCode `bierner.markdown-mermaid`, IntelliJ Markdown 플러그인에서 렌더링)
> 최종 갱신: 2026-05-17

---

## 1. 클래스 다이어그램 — Core 도메인

`FaceRecognitionSystem`이 외부 진입점(Facade)이고, 내부적으로 `FaceDatabase`를 합성하며 DeepFace/FAISS와 협업합니다.

```mermaid
classDiagram
    direction LR

    class FaceRecognitionSystem {
        +str model_name
        +str distance_metric
        +FaceDatabase database
        +__init__(model_name, distance_metric)
        +detect_face(img_path) dict
        +extract_face(img_path, target_size) ndarray
        +represent_face(img_path) list~float~
        +add_face_to_database(img_path, identity) bool
        +recognize_face(img_path, threshold, top_k) list~tuple~
        +recognize_all_faces(img_path, threshold, top_k) list~dict~
        +save_database(file_path) void
        +load_database(file_path) bool
        +visualize_recognition(img_path, threshold, top_k) ndarray
    }

    class FaceDatabase {
        +str model_name
        +str distance_metric
        +list~list~float~~ embeddings
        +list~str~ identities
        +faiss.Index index
        -bool _index_dirty
        +__init__(model_name, distance_metric)
        +add_face(img_path, identity) bool
        +build_index() void
        -_build_index() void
        +search(query_embedding, threshold, top_k) list~tuple~
        +save(file_path) void
        +load(file_path) bool
        -_load_npz_json(npz_path, json_path) bool
        -_load_and_migrate_pkl(pkl_path) bool
    }

    class FaceDetectionFns {
        <<module: face_detection>>
        +detect_face(img, detector_backend) dict
        +extract_face(img, target_size, detector_backend) ndarray
    }

    class FaceRepresentationFns {
        <<module: face_representation>>
        +represent_faces(img, model_name, detector_backend) list~dict~
        +represent_face(img, model_name, detector_backend) list~float~
    }

    class VisualizationFns {
        <<module: visualization>>
        +visualize_recognition(img, face_results) ndarray
    }

    class Validators {
        <<module: validators>>
        +validate_image_path(img_path) void
        +validate_image_input(img) void
        +validate_threshold(threshold) void
        +validate_top_k(top_k) void
        +validate_identity(identity) void
    }

    class Config {
        <<module: config>>
        +bool USE_GPU
        +str DEFAULT_MODEL_NAME
        +str DEFAULT_DISTANCE_METRIC
        +str DEFAULT_DETECTOR_BACKEND
        +float DEFAULT_THRESHOLD
        +int DEFAULT_TOP_K
        +str DATABASE_FILE
        +configure_runtime() void
        +get_logger(name) Logger
    }

    class DeepFace {
        <<external>>
        +represent(img_path, model_name, ...) list~dict~
        +extract_faces(img_path, detector_backend, ...) list~dict~
    }

    class FaissIndex {
        <<external: faiss>>
        +IndexFlatIP
        +IndexFlatL2
        +add(embeddings) void
        +search(query, k) tuple
        +normalize_L2(embeddings) void
    }

    FaceRecognitionSystem *-- FaceDatabase : composes
    FaceRecognitionSystem ..> FaceDetectionFns : uses
    FaceRecognitionSystem ..> FaceRepresentationFns : uses
    FaceRecognitionSystem ..> VisualizationFns : uses
    FaceDatabase ..> FaceRepresentationFns : uses
    FaceDatabase o-- FaissIndex : holds
    FaceDetectionFns ..> DeepFace : delegates
    FaceRepresentationFns ..> DeepFace : delegates
    FaceDetectionFns ..> Validators : validates
    FaceRepresentationFns ..> Validators : validates
    FaceDatabase ..> Validators : validates
    FaceRecognitionSystem ..> Config : reads defaults
```

---

## 2. 클래스 다이어그램 — API 레이어 (FastAPI)

라우터는 `AppState`를 통해 도메인 계층에 접근합니다. Pydantic 스키마가 HTTP 경계의 계약을 정의합니다.

```mermaid
classDiagram
    direction TB

    class AppState {
        <<dataclass>>
        +FaceRecognitionSystem system
        +str detector_backend
        +float default_threshold
        +int default_top_k
        +bool model_loaded
    }

    class FastAPIApp {
        +lifespan(app) AsyncContextManager
        +health() HealthResponse
        +spa_fallback(full_path) FileResponse
    }

    class RecognitionRouter {
        <<router>>
        +recognize(image, threshold, top_k) RecognizeResponse
    }

    class DetectionRouter {
        <<router>>
        +detect(image) DetectResponse
    }

    class DatabaseRouter {
        <<router>>
        +get_status() DatabaseStatus
        +add_face(image, identity) AddFaceResponse
        +rebuild() RebuildResponse
    }

    class SettingsRouter {
        <<router>>
        +get_options() Options
        +get_settings() Settings
        +update_settings(payload) Settings
    }

    class ApiUtils {
        <<module: api.utils>>
        +read_upload_as_bgr(upload) ndarray
        +encode_rgb_to_png_b64(rgb) str
        +encode_bgr_to_png_b64(bgr) str
        +db_status_from_system(system) DatabaseStatus
    }

    class FacialArea {
        <<schema>>
        +int x
        +int y
        +int w
        +int h
    }
    class Match {
        <<schema>>
        +str identity
        +float score
    }
    class FaceResult {
        <<schema>>
        +FacialArea facial_area
        +list~Match~ matches
    }
    class RecognizeResponse {
        <<schema>>
        +list~FaceResult~ faces
        +str? annotated_png_base64
    }
    class DetectResponse {
        <<schema>>
        +FacialArea facial_area
        +float? confidence
        +str detector_backend
        +str? extracted_png_base64
    }
    class DatabaseStatus {
        <<schema>>
        +list~PersonStat~ persons
        +int person_count
        +int total_faces
    }
    class PersonStat {
        <<schema>>
        +str name
        +int face_count
    }
    class AddFaceResponse {
        <<schema>>
        +bool success
        +str message
        +DatabaseStatus db_status
    }
    class RebuildResponse {
        <<schema>>
        +bool success
        +int person_count
        +int face_count
        +DatabaseStatus db_status
    }
    class Settings {
        <<schema>>
        +str model_name
        +Literal distance_metric
        +str detector_backend
        +float threshold
        +int top_k
    }
    class SettingsUpdate {
        <<schema>>
        +str? model_name
        +Literal? distance_metric
        +str? detector_backend
        +float? threshold
        +int? top_k
    }
    class Options {
        <<schema>>
        +list~str~ models
        +list~str~ metrics
        +list~str~ detectors
    }
    class HealthResponse {
        <<schema>>
        +Literal status
        +bool model_loaded
    }

    FastAPIApp o-- AppState : app.state
    FastAPIApp --> RecognitionRouter : include_router
    FastAPIApp --> DetectionRouter : include_router
    FastAPIApp --> DatabaseRouter : include_router
    FastAPIApp --> SettingsRouter : include_router

    RecognitionRouter ..> AppState : reads
    DetectionRouter ..> AppState : reads
    DatabaseRouter ..> AppState : reads/writes
    SettingsRouter ..> AppState : reads/writes

    RecognitionRouter ..> ApiUtils
    DetectionRouter ..> ApiUtils
    DatabaseRouter ..> ApiUtils

    AppState *-- FaceRecognitionSystem

    RecognitionRouter ..> RecognizeResponse : returns
    DetectionRouter ..> DetectResponse : returns
    DatabaseRouter ..> AddFaceResponse : returns
    DatabaseRouter ..> RebuildResponse : returns
    DatabaseRouter ..> DatabaseStatus : returns
    SettingsRouter ..> Settings : returns
    SettingsRouter ..> Options : returns
    FastAPIApp ..> HealthResponse : returns

    FaceResult *-- FacialArea
    FaceResult *-- Match
    RecognizeResponse *-- FaceResult
    DetectResponse *-- FacialArea
    DatabaseStatus *-- PersonStat
    AddFaceResponse *-- DatabaseStatus
    RebuildResponse *-- DatabaseStatus
```

---

## 3. 클래스 다이어그램 — 프론트엔드 (Vue 3 + Pinia)

```mermaid
classDiagram
    direction TB

    class App_vue {
        <<root component>>
        +template
    }

    class AppHeader {
        <<component>>
        +template
    }

    class RouterView {
        <<vue-router>>
    }

    class RecognizeView {
        <<view>>
        -ref file
        -ref result
        +onRecognize()
    }
    class DatabaseView {
        <<view>>
        -ref status
        -ref identity
        +fetchStatus()
        +onAddFace()
        +onRebuild()
    }
    class DetectView {
        <<view>>
        -ref file
        -ref result
        +onDetect()
    }
    class SettingsView {
        <<view>>
        +useSettingsStore
        +onSubmit()
    }

    class ImageUpload {
        <<component>>
        +emit('change', file)
    }

    class SettingsStore {
        <<pinia store>>
        +Ref~Settings~ settings
        +Ref~Options~ options
        +Ref~boolean~ loading
        +Ref~string~ error
        +fetch() Promise
        +update(patch) Promise
    }

    class ApiClient {
        <<axios>>
        +baseURL = '/api'
        +timeout = 60_000
        +extractErrorMessage(err) string
    }

    class RecognitionApi {
        <<module>>
        +recognizeImage(file, th, k) RecognizeResponse
        +detectImage(file) DetectResponse
    }
    class DatabaseApi {
        <<module>>
        +getDatabaseStatus() DatabaseStatus
        +addFace(file, identity) AddFaceResponse
        +rebuildDatabase() RebuildResponse
    }
    class SettingsApi {
        <<module>>
        +getSettings() Settings
        +updateSettings(patch) Settings
        +getOptions() Options
        +getHealth() HealthResponse
    }

    class Router {
        <<vue-router>>
        +/recognize → RecognizeView
        +/database → DatabaseView
        +/detect → DetectView
        +/settings → SettingsView
    }

    App_vue *-- AppHeader
    App_vue *-- RouterView
    Router --> RecognizeView
    Router --> DatabaseView
    Router --> DetectView
    Router --> SettingsView

    RecognizeView ..> ImageUpload
    DatabaseView ..> ImageUpload
    DetectView ..> ImageUpload

    RecognizeView ..> RecognitionApi
    DetectView ..> RecognitionApi
    DatabaseView ..> DatabaseApi
    SettingsView ..> SettingsStore
    SettingsStore ..> SettingsApi

    RecognitionApi ..> ApiClient
    DatabaseApi ..> ApiClient
    SettingsApi ..> ApiClient
```

---

## 4. 시퀀스 다이어그램 — `POST /api/recognize` 전체 흐름

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant View as RecognizeView.vue
    participant API as recognition.ts
    participant HTTP as axios client
    participant FA as FastAPI
    participant R as recognition router
    participant Utl as api.utils
    participant Sys as FaceRecognitionSystem
    participant Rep as face_representation
    participant DF as DeepFace
    participant DB as FaceDatabase
    participant FX as FAISS Index
    participant Viz as visualization

    U->>View: 이미지 선택 + threshold/top_k
    View->>API: recognizeImage(file, th, k)
    API->>HTTP: POST /api/recognize (multipart)
    HTTP->>FA: HTTP 요청
    FA->>R: recognize(image, threshold, top_k)
    R->>Utl: read_upload_as_bgr(image)
    Utl-->>R: bgr ndarray
    R->>R: threshold/top_k 검증 (422 if invalid)
    R->>Sys: recognize_all_faces(bgr, th, k)
    Sys->>Rep: represent_faces(bgr, model_name)
    Rep->>DF: DeepFace.represent(bgr, ...)
    DF-->>Rep: [{embedding, facial_area}, ...]
    Rep-->>Sys: face_objs

    loop 각 얼굴마다
        Sys->>DB: search(embedding, th, k)
        DB->>DB: 인덱스 dirty면 build_index
        DB->>FX: index.search(query, k)
        FX-->>DB: distances, indices
        DB->>DB: threshold 필터링
        DB-->>Sys: list[(identity, score)]
    end
    Sys-->>R: face_results

    R->>Viz: visualize_recognition(bgr, face_results)
    Viz-->>R: annotated_rgb
    R->>Utl: encode_rgb_to_png_b64(annotated_rgb)
    Utl-->>R: base64 PNG

    R-->>FA: RecognizeResponse
    FA-->>HTTP: 200 OK (JSON)
    HTTP-->>API: data
    API-->>View: RecognizeResponse
    View->>U: 결과 + 주석 이미지 표시
```

---

## 5. 시퀀스 다이어그램 — `POST /api/database/faces` (얼굴 등록)

```mermaid
sequenceDiagram
    autonumber
    participant View as DatabaseView.vue
    participant R as database router
    participant Sys as FaceRecognitionSystem
    participant DB as FaceDatabase
    participant Rep as face_representation
    participant Disk as 파일시스템

    View->>R: POST /api/database/faces (image, identity)
    R->>R: identity 공백 제거 + 빈값 검사 (422)
    R->>R: read_upload_as_bgr (5MB 검사)
    R->>Sys: add_face_to_database(bgr, identity)
    Sys->>DB: add_face(bgr, identity)
    DB->>DB: validate_identity
    DB->>Rep: represent_face(bgr, model_name)
    Rep-->>DB: embedding (or None)

    alt embedding != None
        DB->>DB: embeddings.append, identities.append
        DB->>DB: _index_dirty = True
        DB-->>Sys: True
        Sys-->>R: True
        R->>DB: build_index()
        DB->>DB: FAISS IndexFlatIP/L2 재구축
        R->>Sys: save_database(DATABASE_FILE)
        Sys->>DB: save(file_path)
        DB->>Disk: write face_database.npz
        DB->>Disk: write face_database.json
        R-->>View: AddFaceResponse(success=true, db_status)
    else 얼굴 검출 실패
        DB-->>Sys: False
        Sys-->>R: False
        R-->>View: AddFaceResponse(success=false, message)
    end
```

---

## 6. 시퀀스 다이어그램 — 앱 부팅 (Lifespan)

```mermaid
sequenceDiagram
    autonumber
    participant Uv as uvicorn
    participant App as FastAPI lifespan
    participant Cfg as config
    participant State as AppState
    participant Sys as FaceRecognitionSystem
    participant DB as FaceDatabase
    participant Disk as 파일시스템
    participant Rep as face_representation
    participant DF as DeepFace

    Uv->>App: 앱 시작
    App->>App: os.chdir(backend/)
    App->>Cfg: configure_runtime()
    Cfg->>Cfg: CUDA_VISIBLE_DEVICES 설정
    Cfg->>Cfg: tf.config.set_visible_devices
    App->>State: AppState() (system=FaceRecognitionSystem())

    alt face_database.npz + .json 존재
        State->>Sys: load_database(DATABASE_FILE)
        Sys->>DB: load → _load_npz_json
        DB->>Disk: read npz + json
        DB->>DB: model_name 일치 검사 (실패 시 ValueError)
        DB->>DB: _build_index() (FAISS)
    else .pkl만 존재 (레거시)
        State->>Sys: load_database(.pkl)
        Sys->>DB: load → _load_and_migrate_pkl
        DB->>Disk: read pkl
        DB->>DB: _build_index()
        DB->>Disk: save npz + json
        DB->>Disk: remove .pkl
    end

    opt SKIP_WARMUP != 1
        App->>Rep: represent_face(dummy 224×224)
        Rep->>DF: DeepFace.represent (가중치 로드)
        DF-->>Rep: embedding
        App->>State: model_loaded = True
    end

    App->>App: app.state.app_state = state
    Note over App: yield → 요청 처리 준비 완료
```

---

## 7. 시퀀스 다이어그램 — `PUT /api/settings` (모델 변경 시)

```mermaid
sequenceDiagram
    autonumber
    participant View as SettingsView.vue
    participant Store as settings store
    participant API as settings.ts
    participant R as settings router
    participant State as AppState
    participant NewSys as new FaceRecognitionSystem
    participant DB as FaceDatabase

    View->>Store: update({ model_name: 'ArcFace' })
    Store->>API: updateSettings(patch)
    API->>R: PUT /api/settings
    R->>R: 모델/검출기 화이트리스트 검증 (422 if invalid)
    R->>State: detector_backend/threshold/top_k 갱신

    alt 모델 또는 메트릭 변경
        R->>NewSys: FaceRecognitionSystem(new_model, new_metric)
        R->>State: system = NewSys
        opt 기존 DB 파일 존재
            R->>NewSys: load_database(DATABASE_FILE)
            NewSys->>DB: load
            alt model_name 불일치
                DB-->>NewSys: ValueError → False (빈 상태)
            else 일치
                DB-->>NewSys: True (인덱스 재구축됨)
            end
        end
    end

    R-->>API: Settings (현재 상태)
    API-->>Store: Settings
    Store->>Store: settings.value = updated
    Store-->>View: 갱신 완료
```

---

## 8. 상태 다이어그램 — `FaceDatabase` 인덱스 상태

`_index_dirty` 플래그로 lazy rebuild를 구현합니다.

```mermaid
stateDiagram-v2
    [*] --> Empty: __init__

    Empty --> Dirty: add_face (embedding != None)
    Empty --> Built: load (npz+json)
    Empty --> Built: _load_and_migrate_pkl

    Dirty --> Built: build_index() / _build_index()
    Dirty --> Built: search (lazy rebuild)

    Built --> Dirty: add_face

    Built --> [*]: save (npz+json)
    Dirty --> [*]: save (npz+json)

    note right of Dirty
        embeddings/identities 변경됨
        FAISS index는 outdated
    end note

    note right of Built
        FAISS index가 최신
        search 가능
    end note
```

---

## 9. 컴포넌트 다이어그램 — 전체 시스템

```mermaid
flowchart TB
    subgraph Frontend["Frontend (Vue 3 SPA)"]
        direction TB
        Router2["vue-router"]
        Views["Views<br/>(Recognize/Database/Detect/Settings)"]
        Components["Components<br/>(AppHeader, ImageUpload)"]
        Stores["Pinia Stores<br/>(settings)"]
        ApiLayer["API Layer<br/>(axios client + endpoints)"]
        Types["TypeScript Types<br/>(api.ts)"]
    end

    subgraph Backend["Backend (FastAPI)"]
        direction TB
        Lifespan["lifespan<br/>(DB load + warmup)"]
        Routers2["Routers<br/>(recognition/detection/database/settings)"]
        Schemas["Pydantic Schemas"]
        DepsState["AppState (단일 인스턴스)"]
        Facade["FaceRecognitionSystem"]
        DBComp["FaceDatabase"]
        DetMod["face_detection"]
        RepMod["face_representation"]
        VizMod["visualization"]
        ValMod["validators"]
        CfgMod["config"]
    end

    subgraph ExtLibs["External Libraries"]
        DeepFaceLib["DeepFace"]
        FaissLib["faiss-cpu"]
        TFLib["TensorFlow"]
        Cv2Lib["OpenCV"]
    end

    subgraph DiskFS["Disk"]
        NpzFile[("face_database.npz")]
        JsonFile[("face_database.json")]
        ImgDir[("face_database/&lt;person&gt;/*.jpg")]
    end

    Views --> Router2
    Views --> Components
    Views --> Stores
    Views --> ApiLayer
    ApiLayer --> Types
    Stores --> ApiLayer

    ApiLayer -. "HTTP /api/*" .-> Routers2

    Routers2 --> Schemas
    Routers2 --> DepsState
    DepsState --> Facade
    Lifespan --> DepsState
    Facade --> DBComp
    Facade --> DetMod
    Facade --> RepMod
    Facade --> VizMod
    DBComp --> RepMod
    DBComp --> ValMod
    DetMod --> ValMod
    RepMod --> ValMod
    Facade --> CfgMod

    DetMod --> DeepFaceLib
    RepMod --> DeepFaceLib
    DBComp --> FaissLib
    VizMod --> Cv2Lib
    DeepFaceLib --> TFLib

    DBComp <-->|"save/load"| NpzFile
    DBComp <-->|"save/load"| JsonFile
    Facade -. "build_database" .-> ImgDir
```

---

## 10. 유스케이스 다이어그램

```mermaid
flowchart LR
    User((사용자))

    UC1["얼굴 인식<br/>(여러 얼굴 일괄)"]
    UC2["얼굴 검출/추출"]
    UC3["인물 등록<br/>(얼굴 추가)"]
    UC4["DB 재구축<br/>(폴더 일괄 import)"]
    UC5["DB 상태 조회"]
    UC6["설정 변경<br/>(모델/메트릭/임계값)"]
    UC7["헬스체크"]

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
    User --> UC7

    UC1 -. include .-> UCi1["임베딩 추출"]
    UC1 -. include .-> UCi2["FAISS 검색"]
    UC1 -. include .-> UCi3["bbox 시각화"]

    UC3 -. include .-> UCi1
    UC3 -. include .-> UCi4["인덱스 재빌드"]
    UC3 -. include .-> UCi5["DB 저장"]

    UC4 -. include .-> UCi1
    UC4 -. include .-> UCi4
    UC4 -. include .-> UCi5

    UC6 -. extend .-> UCe1["모델 교체 시<br/>DB 재로드 시도"]
```

---

## 11. 클래스 다이어그램 — 입력 검증 모듈

```mermaid
classDiagram
    class Validators {
        <<module>>
        +SUPPORTED_IMAGE_FORMATS: set
        +ImageInput: TypeAlias = str | ndarray
        +validate_image_path(img_path: str) void
        +validate_image_input(img: ImageInput) void
        +validate_threshold(threshold: float) void
        +validate_top_k(top_k: int) void
        +validate_identity(identity: str) void
    }

    class TypeError {
        <<exception>>
    }
    class ValueError {
        <<exception>>
    }
    class FileNotFoundError {
        <<exception>>
    }

    Validators ..> TypeError : raises
    Validators ..> ValueError : raises
    Validators ..> FileNotFoundError : raises

    note for Validators "ImageInput은 str(path) 또는<br/>(H,W,3) BGR ndarray 허용.<br/>API 레이어는 항상 ndarray로 전달."
```

---

## 12. 렌더링 안내

- **GitHub** — `.md` 내 Mermaid 코드블록을 자동 렌더링
- **VSCode** — 확장 `bierner.markdown-mermaid` 설치 후 미리보기
- **IntelliJ/PyCharm** — Markdown 플러그인에서 Mermaid 지원 (Settings → Languages & Frameworks → Markdown → Mermaid 활성화)
- **로컬 변환** — `npx @mermaid-js/mermaid-cli -i docs/uml.md -o docs/uml.svg`
