// 백엔드 Pydantic 스키마와 1:1 대응

export interface FacialArea {
  x: number
  y: number
  w: number
  h: number
}

export interface Match {
  identity: string
  score: number
}

export interface FaceResult {
  facial_area: FacialArea
  matches: Match[]
}

export interface RecognizeResponse {
  faces: FaceResult[]
  annotated_png_base64: string | null
}

export interface DetectResponse {
  facial_area: FacialArea
  confidence: number | null
  detector_backend: string
  extracted_png_base64: string | null
}

export interface PersonStat {
  name: string
  face_count: number
}

export interface DatabaseStatus {
  persons: PersonStat[]
  person_count: number
  total_faces: number
}

export interface AddFaceResponse {
  success: boolean
  message: string
  db_status: DatabaseStatus
}

export interface RebuildResponse {
  success: boolean
  person_count: number
  face_count: number
  db_status: DatabaseStatus
}

export type DistanceMetric = 'cosine' | 'euclidean' | 'euclidean_l2'

export interface Settings {
  model_name: string
  distance_metric: DistanceMetric
  detector_backend: string
  threshold: number
  top_k: number
}

export type SettingsUpdate = Partial<Settings>

export interface Options {
  models: string[]
  metrics: DistanceMetric[]
  detectors: string[]
}

export interface HealthResponse {
  status: 'ok' | 'loading'
  model_loaded: boolean
}
