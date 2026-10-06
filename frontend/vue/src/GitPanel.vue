<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api, type GitPanelData } from "../../src/api";
import Modal from "./Modal.vue";
const props = defineProps<{ id: string }>();
const emit = defineEmits<{ close: [] }>();
const data = ref<GitPanelData | null>(null), selectedPath = ref(""), diff = ref("");
const message = ref(""), branch = ref(""), initialBranch = ref("main"), busy = ref(false), notice = ref("");
const eligible = computed(() => new Set(data.value?.commit_eligible || []));
const commitPaths = computed(() => (data.value?.status?.changes || []).map(line => line.slice(3).split(" -> ").pop() || line).filter(path => eligible.value.has(path)));
async function load() {
  data.value = await api.git(props.id);
  if (data.value.initialized && !data.value.has_commits) notice.value = "当前仓库尚无提交；首次提交后分支将正式建立。";
}
async function showDiff(path: string) { selectedPath.value = path; try { diff.value = (await api.git(props.id, path)).diff || ""; } catch (e) { notice.value = (e as Error).message; } }
async function act(body: Record<string, unknown>, success: string) {
  const action = body.action as string;
  if (data.value?.has_commits === false && (action === "create_branch" || action === "switch_branch")) { notice.value = "请先完成首次提交，再创建或切换分支。"; return; }
  busy.value = true; notice.value = "";
  try { const result = await api.gitAction(props.id, body); message.value = ""; branch.value = ""; await load(); const hash = typeof result.commit === "string" ? result.commit.slice(0, 12) : ""; notice.value = action === "commit" ? `提交成功 · ${hash} · ${Number((body.paths as string[] || []).length)} 个文件` : success; }
  catch (e) { notice.value = (e as Error).message; } finally { busy.value = false; }
}
onMounted(() => void load().catch(e => { notice.value = e.message; }));
</script>
<template>
  <div class="modal git-modal" @mousedown.self="!busy && emit('close')">
    <section v-if="data && !data.initialized" class="git-card git-init-card"><header><div><small>Git 工作区</small><h2>初始化仓库</h2></div><button @click="emit('close')">×</button></header><main class="git-init"><h3>这个文件夹还不是 Git 仓库</h3><p>{{ data.repo_root }}</p><p>初始化只会创建 <code>.git</code> 元数据，不会自动提交、删除或修改现有文件。</p><label>默认分支名<input v-model="initialBranch" placeholder="main" /></label><button class="primary" :disabled="busy || !initialBranch.trim()" @click="act({ action: 'initialize', branch: initialBranch.trim() }, 'Git 仓库初始化成功')">初始化 Git 仓库</button><p v-if="notice" class="notice">{{ notice }}</p></main></section>
    <section v-else class="git-card"><header><div><small>Git 工作区</small><h2>{{ data?.branches?.current || data?.status?.branch || '读取中…' }}</h2></div><nav><button :disabled="busy" @click="load">↻ 刷新</button><button @click="emit('close')">×</button></nav></header><div class="git-grid"><section class="git-changes"><h3>工作区更改 <b>{{ data?.status?.changes.length || 0 }}</b></h3><div><button v-for="line in data?.status?.changes || []" :key="line" :class="{ active: selectedPath === (line.slice(3).split(' -> ').pop() || line) }" @click="showDiff(line.slice(3).split(' -> ').pop() || line)"><i>{{ line.slice(0, 2) }}</i><span>{{ line.slice(3).split(' -> ').pop() || line }}</span><em v-if="eligible.has(line.slice(3).split(' -> ').pop() || line)">可提交</em></button><p v-if="!data?.status?.changes.length" class="git-empty">工作区干净</p></div><label>提交说明<input v-model="message" placeholder="feat: 描述本次修改" /></label><button class="primary git-commit" :disabled="busy || !message.trim() || !commitPaths.length" @click="act({ action: 'commit', message, paths: commitPaths }, '提交成功')">提交 RAgent 本轮修改 ({{ commitPaths.length }})</button></section><section class="git-diff"><h3>{{ selectedPath || '选择文件查看 Diff' }}</h3><pre class="vue-code">{{ diff || '暂无 Diff' }}</pre></section><aside class="git-meta"><h3>分支</h3><select :value="data?.branches?.current || ''" :disabled="busy" @change="act({ action: 'switch_branch', branch: ($event.target as HTMLSelectElement).value }, `已切换到 ${($event.target as HTMLSelectElement).value}`)"><option v-for="name in data?.branches?.branches || []" :key="name">{{ name }}</option></select><div class="git-new-branch"><input v-model="branch" placeholder="feature/name" /><button :disabled="busy || !branch.trim()" @click="act({ action: 'create_branch', branch }, `已创建并切换到 ${branch}`)">创建</button></div><h3>最近提交</h3><div class="git-log"><article v-for="item in data?.commits || []" :key="item.sha"><code>{{ item.sha }}</code><strong>{{ item.subject }}</strong><small>{{ item.author }} · {{ new Date(item.date).toLocaleDateString() }}</small></article></div></aside></div><footer v-if="notice" class="git-notice">{{ notice }}</footer></section>
  </div>
</template>
