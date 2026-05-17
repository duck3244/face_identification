<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { getHealth } from '@/api/settings'

const health = ref<'ok' | 'loading' | 'unknown'>('unknown')

onMounted(async () => {
  try {
    const h = await getHealth()
    health.value = h.status
  } catch {
    health.value = 'unknown'
  }
})

const links = [
  { to: '/recognize', label: '얼굴 인식' },
  { to: '/database', label: 'DB 관리' },
  { to: '/detect', label: '검출/추출' },
  { to: '/settings', label: '설정' },
]
</script>

<template>
  <header class="bg-white border-b border-slate-200">
    <div class="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
      <h1 class="text-xl font-semibold text-slate-800">얼굴 인식 시스템</h1>
      <nav class="flex gap-1">
        <RouterLink
          v-for="l in links"
          :key="l.to"
          :to="l.to"
          class="px-3 py-1.5 rounded text-sm text-slate-600 hover:bg-slate-100"
          active-class="bg-blue-50 text-blue-700 font-medium"
        >
          {{ l.label }}
        </RouterLink>
      </nav>
      <span
        class="text-xs px-2 py-1 rounded-full"
        :class="{
          'bg-green-100 text-green-700': health === 'ok',
          'bg-amber-100 text-amber-700': health === 'loading',
          'bg-slate-100 text-slate-500': health === 'unknown',
        }"
      >
        {{ health === 'ok' ? '준비됨' : health === 'loading' ? '로딩 중' : '연결 안 됨' }}
      </span>
    </div>
  </header>
</template>
