<script setup lang="ts">
import { ref, onMounted } from 'vue'
import ImageUpload from '@/components/ImageUpload.vue'
import { getDatabaseStatus, addFace, rebuildDatabase } from '@/api/database'
import { extractErrorMessage } from '@/api/client'
import type { DatabaseStatus } from '@/types/api'

const status = ref<DatabaseStatus | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const message = ref<string | null>(null)

const addFile = ref<File | null>(null)
const addName = ref('')
const adding = ref(false)
const rebuilding = ref(false)

async function refresh() {
  loading.value = true
  error.value = null
  try {
    status.value = await getDatabaseStatus()
  } catch (e) {
    error.value = extractErrorMessage(e)
  } finally {
    loading.value = false
  }
}

async function onAdd() {
  message.value = null
  error.value = null
  if (!addFile.value || !addName.value.trim()) {
    error.value = '이미지와 인물 이름을 모두 입력해주세요.'
    return
  }
  adding.value = true
  try {
    const r = await addFace(addFile.value, addName.value.trim())
    message.value = r.message
    status.value = r.db_status
    addFile.value = null
    addName.value = ''
  } catch (e) {
    error.value = extractErrorMessage(e)
  } finally {
    adding.value = false
  }
}

async function onRebuild() {
  if (!confirm('face_database/ 디렉토리에서 DB를 전면 재구축합니다. 진행할까요?')) return
  message.value = null
  error.value = null
  rebuilding.value = true
  try {
    const r = await rebuildDatabase()
    message.value = `재구축 완료: 인물 ${r.person_count}명 / 얼굴 ${r.face_count}개`
    status.value = r.db_status
  } catch (e) {
    error.value = extractErrorMessage(e)
  } finally {
    rebuilding.value = false
  }
}

onMounted(refresh)
</script>

<template>
  <div class="space-y-6">
    <section class="bg-white rounded-lg shadow-sm p-5">
      <div class="flex items-center justify-between mb-3">
        <h2 class="text-lg font-semibold text-slate-800">데이터베이스 상태</h2>
        <button class="text-sm text-blue-600 hover:underline" :disabled="loading" @click="refresh">새로고침</button>
      </div>
      <div v-if="loading" class="text-slate-400 py-4 text-center">로딩...</div>
      <div v-else-if="status">
        <p class="text-sm text-slate-600 mb-3">
          인물 수: <strong>{{ status.person_count }}명</strong>
          / 총 얼굴 수: <strong>{{ status.total_faces }}개</strong>
        </p>
        <table class="w-full text-sm">
          <thead class="text-left text-slate-500 border-b">
            <tr>
              <th class="py-2">인물</th>
              <th class="py-2 text-right">얼굴 수</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="p in status.persons" :key="p.name" class="border-b border-slate-100">
              <td class="py-1.5 text-slate-800">{{ p.name }}</td>
              <td class="py-1.5 text-right tabular-nums">{{ p.face_count }}</td>
            </tr>
            <tr v-if="status.persons.length === 0">
              <td colspan="2" class="py-4 text-center text-slate-400">등록된 인물이 없습니다.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="bg-white rounded-lg shadow-sm p-5 space-y-4">
      <h2 class="text-lg font-semibold text-slate-800">새 얼굴 추가</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ImageUpload label="얼굴 이미지" @change="(f: File | null) => (addFile = f)" />
        <div class="space-y-3">
          <div>
            <label class="block text-sm font-medium text-slate-700 mb-1">인물 이름</label>
            <input
              v-model="addName"
              type="text"
              placeholder="예: John Doe"
              class="w-full px-3 py-2 border border-slate-300 rounded"
            />
          </div>
          <button
            class="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium py-2 rounded"
            :disabled="adding || !addFile || !addName.trim()"
            @click="onAdd"
          >
            {{ adding ? '추가 중...' : '얼굴 추가' }}
          </button>
        </div>
      </div>
    </section>

    <section class="bg-white rounded-lg shadow-sm p-5">
      <h2 class="text-lg font-semibold text-slate-800 mb-3">전체 재구축</h2>
      <p class="text-sm text-slate-500 mb-3">
        backend/face_database/ 의 인물 디렉토리에서 모든 얼굴을 다시 등록합니다.
      </p>
      <button
        class="bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white font-medium px-4 py-2 rounded"
        :disabled="rebuilding"
        @click="onRebuild"
      >
        {{ rebuilding ? '재구축 중...' : '데이터베이스 재구축' }}
      </button>
    </section>

    <p v-if="message" class="text-sm text-green-700 bg-green-50 border border-green-200 rounded p-3">{{ message }}</p>
    <p v-if="error" class="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-3">{{ error }}</p>
  </div>
</template>
