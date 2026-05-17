import { client } from './client'
import type { Options, Settings, SettingsUpdate, HealthResponse } from '@/types/api'

export async function getSettings(): Promise<Settings> {
  const { data } = await client.get<Settings>('/settings')
  return data
}

export async function updateSettings(payload: SettingsUpdate): Promise<Settings> {
  const { data } = await client.put<Settings>('/settings', payload)
  return data
}

export async function getOptions(): Promise<Options> {
  const { data } = await client.get<Options>('/options')
  return data
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>('/health')
  return data
}
