<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import { extractErrorMessage } from '@/api/client'
import type { Settings } from '@/types/api'

const store = useSettingsStore()
const form = ref<Settings | null>(null)
const saving = ref(false)
const message = ref<string | null>(null)
const error = ref<string | null>(null)

onMounted(async () => {
  if (!store.settings) await store.fetch()
  if (store.settings) form.value = { ...store.settings }
})

watch(
  () => store.settings,
  (s) => {
    if (s && !form.value) form.value = { ...s }
  },
)

async function onSave() {
  if (!form.value) return
  saving.value = true
  message.value = null
  error.value = null
  try {
    const updated = await store.update(form.value)
    form.value = { ...updated }
    message.value = '설정이 적용되었습니다.'
  } catch (e) {
    error.value = extractErrorMessage(e)
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <section class="bg-white rounded-lg shadow-sm p-5 space-y-4 max-w-2xl">
    <h2 class="text-lg font-semibold text-slate-800">시스템 설정</h2>
    <p class="text-sm text-slate-500">
      모델/메트릭을 변경하면 기존 DB가 호환되지 않을 수 있습니다. 변경 후 DB 재구축이 필요할 수 있습니다.
    </p>

    <div v-if="store.loading || !form" class="py-8 text-center text-slate-400">로딩 중...</div>
    <div v-else class="space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <label class="block">
          <span class="text-sm font-medium text-slate-700">인식 모델</span>
          <select v-model="form.model_name" class="mt-1 w-full px-2 py-1.5 border border-slate-300 rounded">
            <option v-for="m in store.options?.models ?? []" :key="m" :value="m">{{ m }}</option>
          </select>
        </label>
        <label class="block">
          <span class="text-sm font-medium text-slate-700">거리 메트릭</span>
          <select v-model="form.distance_metric" class="mt-1 w-full px-2 py-1.5 border border-slate-300 rounded">
            <option v-for="m in store.options?.metrics ?? []" :key="m" :value="m">{{ m }}</option>
          </select>
        </label>
        <label class="block">
          <span class="text-sm font-medium text-slate-700">검출 백엔드</span>
          <select v-model="form.detector_backend" class="mt-1 w-full px-2 py-1.5 border border-slate-300 rounded">
            <option v-for="d in store.options?.detectors ?? []" :key="d" :value="d">{{ d }}</option>
          </select>
        </label>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium text-slate-700">
            기본 임계값: {{ form.threshold.toFixed(2) }}
          </label>
          <input v-model.number="form.threshold" type="range" min="0" max="1" step="0.05" class="w-full" />
        </div>
        <div>
          <label class="block text-sm font-medium text-slate-700">기본 Top-K</label>
          <input
            v-model.number="form.top_k"
            type="number"
            min="1"
            max="20"
            class="w-full px-2 py-1.5 border border-slate-300 rounded"
          />
        </div>
      </div>
      <button
        class="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium px-4 py-2 rounded"
        :disabled="saving"
        @click="onSave"
      >
        {{ saving ? '적용 중...' : '설정 적용' }}
      </button>
      <p v-if="message" class="text-sm text-green-700 bg-green-50 border border-green-200 rounded p-3">{{ message }}</p>
      <p v-if="error" class="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-3">{{ error }}</p>
    </div>
  </section>
</template>
