import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Options, Settings } from '@/types/api'
import { getOptions, getSettings, updateSettings as apiUpdate } from '@/api/settings'

export const useSettingsStore = defineStore('settings', () => {
  const settings = ref<Settings | null>(null)
  const options = ref<Options | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetch() {
    loading.value = true
    error.value = null
    try {
      const [s, o] = await Promise.all([getSettings(), getOptions()])
      settings.value = s
      options.value = o
    } catch (e) {
      error.value = String(e)
    } finally {
      loading.value = false
    }
  }

  async function update(patch: Partial<Settings>) {
    const updated = await apiUpdate(patch)
    settings.value = updated
    return updated
  }

  return { settings, options, loading, error, fetch, update }
})
