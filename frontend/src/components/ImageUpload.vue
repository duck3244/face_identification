<script setup lang="ts">
import { ref, onBeforeUnmount, watch } from 'vue'

const props = defineProps<{ label?: string }>()
const emit = defineEmits<{ (e: 'change', file: File | null): void }>()

const previewUrl = ref<string | null>(null)
const fileName = ref<string | null>(null)
const isDragging = ref(false)
const inputEl = ref<HTMLInputElement | null>(null)

function revoke() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
}

function setFile(file: File | null) {
  revoke()
  if (file) {
    previewUrl.value = URL.createObjectURL(file)
    fileName.value = file.name
  } else {
    fileName.value = null
  }
  emit('change', file)
}

function onFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  setFile(target.files?.[0] ?? null)
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  isDragging.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f && f.type.startsWith('image/')) setFile(f)
}

function clear() {
  setFile(null)
  if (inputEl.value) inputEl.value.value = ''
}

watch(previewUrl, () => {}, { immediate: false })
onBeforeUnmount(revoke)
</script>

<template>
  <div>
    <label v-if="props.label" class="block text-sm font-medium text-slate-700 mb-1">{{ props.label }}</label>
    <div
      class="relative border-2 border-dashed rounded-lg p-4 transition-colors"
      :class="isDragging ? 'border-blue-400 bg-blue-50' : 'border-slate-300 bg-white'"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop="onDrop"
    >
      <input
        ref="inputEl"
        type="file"
        accept="image/*"
        class="absolute inset-0 opacity-0 cursor-pointer"
        @change="onFileChange"
      />
      <div v-if="!previewUrl" class="text-center text-slate-500 py-6 pointer-events-none">
        <p class="text-sm">이미지를 드래그하거나 클릭해서 업로드</p>
        <p class="text-xs mt-1">PNG/JPG/JPEG · 최대 5MB</p>
      </div>
      <div v-else class="space-y-2">
        <img :src="previewUrl" alt="preview" class="max-h-96 mx-auto rounded" />
        <div class="flex justify-between items-center text-xs text-slate-500">
          <span class="truncate">{{ fileName }}</span>
          <button
            type="button"
            class="text-red-600 hover:underline relative z-10"
            @click.stop="clear"
          >
            제거
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
