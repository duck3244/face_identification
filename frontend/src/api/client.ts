import axios from 'axios'

export const client = axios.create({
  baseURL: '/api',
  timeout: 60_000,
})

export function extractErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((d) => d.msg ?? JSON.stringify(d)).join(', ')
    return err.message
  }
  return String(err)
}
