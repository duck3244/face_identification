<script setup lang="ts">
import { ref } from 'vue'
import ImageUpload from '@/components/ImageUpload.vue'
import { detectImage } from '@/api/recognition'
import { extractErrorMessage } from '@/api/client'
import type { DetectResponse } from '@/types/api'

const file = ref<File | null>(null)
const result = ref<DetectResponse | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

async function onDetect() {
  if (!file.value) {
    error.value = '이미지를 업로드해주세요.'
    return
  }
  loading.value = true
  error.value = null
  result.value = null
  try {
    result.value = await detectImage(file.value)
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
      <button
        class="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium py-2 rounded"
        :disabled="loading || !file"
        @click="onDetect"
      >
        {{ loading ? '검출 중...' : '얼굴 검출' }}
      </button>
      <p v-if="error" class="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">{{ error }}</p>
    </section>

    <section class="bg-white rounded-lg shadow-sm p-5 space-y-3">
      <h2 class="text-lg font-semibold text-slate-800">결과</h2>
      <div v-if="!result" class="text-center text-slate-400 py-12">검출 결과가 여기에 표시됩니다.</div>
      <div v-else class="space-y-3">
        <div v-if="result.extracted_png_base64" class="text-center">
          <p class="text-xs text-slate-500 mb-1">추출된 얼굴</p>
          <img
            :src="`data:image/png;base64,${result.extracted_png_base64}`"
            alt="face"
            class="mx-auto rounded border border-slate-200 max-h-64"
          />
        </div>
        <dl class="text-sm text-slate-700 grid grid-cols-2 gap-1 bg-slate-50 p-3 rounded">
          <dt class="text-slate-500">위치 x, y</dt>
          <dd class="tabular-nums">{{ result.facial_area.x }}, {{ result.facial_area.y }}</dd>
          <dt class="text-slate-500">크기 w × h</dt>
          <dd class="tabular-nums">{{ result.facial_area.w }} × {{ result.facial_area.h }}</dd>
          <dt class="text-slate-500">신뢰도</dt>
          <dd class="tabular-nums">{{ result.confidence !== null ? result.confidence.toFixed(4) : 'N/A' }}</dd>
          <dt class="text-slate-500">검출 백엔드</dt>
          <dd>{{ result.detector_backend }}</dd>
        </dl>
      </div>
    </section>
  </div>
</template>
