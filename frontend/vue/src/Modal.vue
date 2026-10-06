<script setup lang="ts">
defineProps<{ title: string; busy?: boolean; wide?: boolean }>();
const emit = defineEmits<{ close: [] }>();
</script>
<template>
  <div class="modal" @mousedown.self="!busy && emit('close')">
    <section class="modal-card vue-form" :class="{ 'vue-wide': wide }" role="dialog" aria-modal="true" :aria-label="title">
      <header class="modal-head"><div><small>RAgent workspace</small><h2>{{ title }}</h2></div><button type="button" :disabled="busy" aria-label="关闭" @click="emit('close')">×</button></header>
      <slot />
    </section>
  </div>
</template>
