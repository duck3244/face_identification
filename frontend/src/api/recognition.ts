import { client } from './client'
import type { RecognizeResponse, DetectResponse } from '@/types/api'

export async function recognizeImage(
  file: File,
  threshold?: number,
  topK?: number,
): Promise<RecognizeResponse> {
  const form = new FormData()
  form.append('image', file)
  if (threshold !== undefined) form.append('threshold', String(threshold))
  if (topK !== undefined) form.append('top_k', String(topK))
  const { data } = await client.post<RecognizeResponse>('/recognize', form)
  return data
}

export async function detectImage(file: File): Promise<DetectResponse> {
  const form = new FormData()
  form.append('image', file)
  const { data } = await client.post<DetectResponse>('/detect', form)
  return data
}
