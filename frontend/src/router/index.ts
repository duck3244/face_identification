import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/recognize' },
  {
    path: '/recognize',
    name: 'recognize',
    component: () => import('@/views/RecognizeView.vue'),
    meta: { title: '얼굴 인식' },
  },
  {
    path: '/database',
    name: 'database',
    component: () => import('@/views/DatabaseView.vue'),
    meta: { title: '데이터베이스' },
  },
  {
    path: '/detect',
    name: 'detect',
    component: () => import('@/views/DetectView.vue'),
    meta: { title: '검출/추출' },
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/views/SettingsView.vue'),
    meta: { title: '설정' },
  },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
