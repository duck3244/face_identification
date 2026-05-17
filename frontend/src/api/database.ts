import { client } from './client'
import type { AddFaceResponse, DatabaseStatus, RebuildResponse } from '@/types/api'

export async function getDatabaseStatus(): Promise<DatabaseStatus> {
  const { data } = await client.get<DatabaseStatus>('/database')
  return data
}

export async function addFace(file: File, identity: string): Promise<AddFaceResponse> {
  const form = new FormData()
  form.append('image', file)
  form.append('identity', identity)
  const { data } = await client.post<AddFaceResponse>('/database/faces', form)
  return data
}

export async function rebuildDatabase(): Promise<RebuildResponse> {
  const { data } = await client.post<RebuildResponse>('/database/rebuild')
  return data
}
