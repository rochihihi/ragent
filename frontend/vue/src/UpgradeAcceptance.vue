<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api, type UpgradeCheck } from "../../src/api";
import Modal from "./Modal.vue";
const emit = defineEmits<{ close: []; back: [] }>();
const version = ref("3.0.0"), checks = ref<UpgradeCheck[]>([]), running = ref<string | null>(null), error = ref("");
const passed = computed(() => checks.value.filter(item => item.status === "passed").length);
async function load() { const result = await api.upgradeChecks(); version.value = result.version; checks.value = result.checks; }
async function runOne(id: string) {
  running.value = id; error.value = ""; checks.value = checks.value.map(item => item.id === id ? { ...item, status: "running" } : item);
  try { const result = await api.runUpgradeCheck(id) as UpgradeCheck; checks.value = checks.value.map(item => item.id === id ? result : item); }
  catch (e) { error.value = (e as Error).message; checks.value = checks.value.map(item => item.id === id ? { ...item, status: "failed", error: error.value } : item); }
  finally { running.value = null; }
}
async function runAll() {
  running.value = "all"; error.value = ""; checks.value = checks.value.map(item => ({ ...item, status: "running" }));
  try { const result = await api.runUpgradeCheck("all") as { version: string; results: UpgradeCheck[] }; version.value = result.version; checks.value = result.results; }
  catch (e) { error.value = (e as Error).message; } finally { running.value = null; }
}
onMounted(() => void load().catch(e => { error.value = e.message; }));
</script>
<template>
  <Modal title="升级验收" @close="emit('close')">
    <section class="acceptance-hero"><div><small>RAgent {{ version }}</small><strong>{{ checks.length ? `${passed} / ${checks.length} 项通过` : '正在读取能力清单' }}</strong><p>每项测试直接调用当前版本的生产代码，不消耗模型 Token。</p></div><button class="primary" :disabled="Boolean(running) || !checks.length" @click="runAll">{{ running === 'all' ? '正在全部验收…' : '全部验收' }}</button></section>
    <div class="acceptance-list"><article v-for="check in checks" :key="check.id" class="acceptance-check" :class="check.status"><div><header><strong>{{ check.title }}</strong><span>{{ check.status === 'passed' ? '已通过' : check.status === 'failed' ? '未通过' : check.status === 'running' ? '测试中' : '未测试' }}</span></header><p>{{ check.description }}</p><small v-for="line in check.evidence || []" :key="line">{{ line }}</small><em v-if="check.error">{{ check.error }}</em></div><button :disabled="Boolean(running)" @click="runOne(check.id)">{{ running === check.id ? '测试中…' : check.status === 'untested' ? '开始测试' : '重新测试' }}</button></article></div>
    <p v-if="error" class="notice">{{ error }}</p>
    <div class="form-actions"><button @click="emit('back')">返回设置</button><button class="primary" @click="emit('close')">完成</button></div>
  </Modal>
</template>
