<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import Modal from "./Modal.vue";

type Skill = { name: string; description: string; content: string };
const props = defineProps<{ id: string }>();
const emit = defineEmits<{ close: [] }>();
const items = ref<Skill[]>([]), saved = ref<string[]>([]), enabled = ref<string[]>([]);
const content = ref(""), error = ref(""), busy = ref(false);
const dirty = computed(() => [...enabled.value].sort().join("|") !== [...saved.value].sort().join("|"));
async function request(method = "GET", body?: object) {
  const response = await fetch(`/studio-api/sessions/${encodeURIComponent(props.id)}/skills`, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "技能请求失败");
  return data as { items: Skill[]; enabled: string[] };
}
async function load(preserve = false) {
  const data = await request(); items.value = data.items; saved.value = data.enabled;
  if (!preserve) enabled.value = [...data.enabled];
}
async function perform(action: () => Promise<void>) {
  busy.value = true; error.value = "";
  try { await action(); } catch (e) { error.value = (e as Error).message; } finally { busy.value = false; }
}
function close() {
  if (dirty.value) { error.value = "启用项尚未保存，请保存或恢复原勾选后关闭。"; return; }
  emit("close");
}
onMounted(() => void load().catch(e => { error.value = e.message; }));
</script>
<template>
  <Modal title="项目技能" :busy="busy" :wide="true" @close="close">
    <p>导入 SKILL.md 后，勾选本会话启用。仅导入说明文件，不执行脚本，请只使用可信内容。</p>
    <p>存放位置：项目的 .agents/skills/技能名/SKILL.md</p>
    <p>已保存启用：{{ saved.length ? saved.join('、') : '无（导入不等于启用）' }}<span v-if="dirty">；当前勾选尚未保存</span></p>
    <p v-if="!items.length">还没有技能。可以导入文件或粘贴完整内容。</p>
    <div v-for="item in items" :key="item.name">
      <label class="vue-check"><input type="checkbox" :disabled="busy" :checked="enabled.includes(item.name)" @change="($event.target as HTMLInputElement).checked ? enabled.push(item.name) : enabled = enabled.filter(name => name !== item.name)" />{{ item.name }}：{{ item.description }}</label>
      <details><summary>查看说明</summary><pre class="vue-code">{{ item.content }}</pre></details>
    </div>
    <label>选择 SKILL.md<input type="file" accept=".md" :disabled="busy" @change="async ($event) => { const file = ($event.target as HTMLInputElement).files?.[0]; if (!file) return; if (file.size > 48000) { error = '文件过大，请使用不超过 12000 字符的技能'; return; } content = await file.text(); }" /></label>
    <label>技能内容<textarea v-model="content" rows="9" maxlength="12000" placeholder="---\nname: my-skill\ndescription: 适用场景\n---\n具体工作流程" /></label>
    <p v-if="error" class="form-error" role="alert">{{ error }}</p>
    <div class="form-actions"><button :disabled="busy" @click="close">关闭</button><button :disabled="busy || !content.trim()" @click="perform(async () => { await request('POST', { content }); content = ''; await load(true); })">导入技能</button><button class="primary" :disabled="busy" @click="perform(async () => { const result = await request('PUT', { names: enabled }); saved = result.enabled; emit('close'); })">保存启用项</button></div>
  </Modal>
</template>
