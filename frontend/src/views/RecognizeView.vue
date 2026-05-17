<script setup lang="ts">
import { ref, onMounted } from 'vue'
import ImageUpload from '@/components/ImageUpload.vue'
import { recognizeImage } from '@/api/recognition'
import { extractErrorMessage } from '@/api/client'
import { useSettingsStore } from '@/stores/settings'
import type { RecognizeResponse } from '@/types/api'

const settings = useSettingsStore()
const file = ref<File | null>(null)
const threshold = ref(0.5)
const topK = ref(3)
const result = ref<RecognizeResponse | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

onMounted(async () => {
  if (!settings.settings) await settings.fetch()
  if (settings.settings) {
    threshold.value = settings.settings.threshold
    topK.value = settings.settings.top_k
  }
})

async function onRecognize() {
  if (!file.value) {
    error.value = '이미지를 업로드해주세요.'
    return
  }
  loading.value = true
  error.value = null
  result.value = null
  try {
    result.value = await recognizeImage(file.value, threshold.value, topK.value)
  } catch (e) {
    error.value = extractErrorMessage(e)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <section class="bg-white rounded-lg shadow-sm p-5 space-y-4">
      <h2 class="text-lg font-semibold text-slate-800">입력</h2>
      <ImageUpload label="이미지" @change="(f: File | null) => (file = f)" />
      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium text-slate-700">임계값: {{ threshold.toFixed(2) }}</label>
          <input v-model.number="threshold" type="range" min="0" max="1" step="0.05" class="w-full" />
        </div>
        <div>
          <label class="block text-sm font-medium text-slate-700">Top-K</label>
          <input v-model.number="topK" type="number" min="1" max="20" class="w-full px-2 py-1 border border-slate-300 rounded" />
        </div>
      </div>
      <button
        type="button"
        class="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium py-2 rounded transition"
        :disabled="loading || !file"
        @click="onRecognize"
      >
        <span v-if="loading">인식 중...</span>
        <span v-else>인식 실행</span>
      </button>
      <p v-if="error" class="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">{{ error }}</p>
    </section>

    <section class="bg-white rounded-lg shadow-sm p-5 space-y-4">
      <h2 class="text-lg font-semibold text-slate-800">결과</h2>
      <div v-if="!result" class="text-center text-slate-400 py-12">인식 결과가 여기에 표시됩니다.</div>
      <div v-else class="space-y-3">
        <img
          v-if="result.annotated_png_base64"
          :src="`data:image/png;base64,${result.annotated_png_base64}`"
          alt="annotated"
          class="max-h-96 mx-auto rounded border border-slate-200"
        />
        <div v-if="result.faces.length === 0" class="text-sm text-slate-500 text-center py-4">
          검출된 얼굴이 없습니다.
        </div>
        <table v-else class="w-full text-sm">
          <thead class="text-left text-slate-500 border-b">
            <tr>
              <th class="py-2 pr-2 w-16">얼굴 #</th>
              <th class="py-2 pr-2">인물</th>
              <th class="py-2 text-right">유사도</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="(face, fi) in result.faces" :key="fi">
              <tr v-if="face.matches.length === 0" class="border-b border-slate-100">
                <td class="py-1.5 pr-2 text-slate-500">{{ fi + 1 }}</td>
                <td class="py-1.5 pr-2 text-slate-400 italic">일치 없음</td>
                <td class="py-1.5 text-right text-slate-300">-</td>
              </tr>
              <tr
                v-for="(m, mi) in face.matches"
                :key="`${fi}-${mi}`"
                class="border-b border-slate-100"
              >
                <td class="py-1.5 pr-2 text-slate-500">{{ mi === 0 ? fi + 1 : '' }}</td>
                <td class="py-1.5 pr-2 text-slate-800 font-medium">{{ m.identity }}</td>
                <td class="py-1.5 text-right tabular-nums">{{ m.score.toFixed(4) }}</td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>
