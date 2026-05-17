<script setup lang="ts">
import AppHeader from './components/AppHeader.vue'
</script>

<template>
  <div class="min-h-screen flex flex-col">
    <AppHeader />
    <main class="flex-1 max-w-6xl w-full mx-auto px-6 py-6">
      <RouterView />
    </main>
  </div>
</template>
