<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { api, type Event, type Provider, type Session } from "../../src/api";
import { useWorkspace, providers, efforts, phases } from "./useWorkspace";
import Modal from "./Modal.vue";
import MessageContent from "./MessageContent.vue";
import FileTree from "./FileTree.vue";
import PermissionDialog from "./PermissionDialog.vue";
import Skills from "./Skills.vue";
import McpSettings from "./McpSettings.vue";
import GitPanel from "./GitPanel.vue";
import ApiSettings from "./ApiSettings.vue";
import FolderPicker from "./FolderPicker.vue";
import UpgradeAcceptance from "./UpgradeAcceptance.vue";
import SandboxSettings from "./SandboxSettings.vue";

const ws = useWorkspace();
const modal = ref<"new" | "folders" | "settings" | "file" | "api" | "api-config" | "skills" | "mcp" | "git" | "acceptance" | "sandbox" | null>(null);
const fileView = ref<"diff" | "before" | "after">("after");
const showSidebar = ref(true), showTrace = ref(true), showAllEvents = ref(false);
const leftPane = ref(270), rightPane = ref(330);
const selectedDirectory = ref("");
const newEntryKind = ref<"file" | "folder" | null>(null);
const newEntryPath = ref("");
const workspaceStyle = computed(() => ({
  "--left-pane": `${leftPane.value}px`,
  "--right-pane": `${rightPane.value}px`,
}));

function resizePane(side: "left" | "right", event: PointerEvent) {
  const handle = event.currentTarget as HTMLElement;
  const workspace = handle.parentElement?.getBoundingClientRect();
  if (!workspace) return;
  handle.setPointerCapture?.(event.pointerId);
  const move = (next: PointerEvent) => {
    if (side === "left") {
      leftPane.value = Math.min(440, Math.max(210, next.clientX - workspace.left));
    } else {
      rightPane.value = Math.min(520, Math.max(270, workspace.right - next.clientX));
    }
  };
  const stop = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", stop);
    window.removeEventListener("pointercancel", stop);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", stop, { once: true });
  window.addEventListener("pointercancel", stop, { once: true });
}

function beginEntry(kind: "file" | "folder") {
  newEntryKind.value = kind;
  newEntryPath.value = "";
}

function windowAction(action: "minimize" | "maximize" | "close") {
  window.pywebview?.api?.[action]?.();
}

async function createEntry() {
  const id = ws.activeId.value;
  const name = newEntryPath.value.trim().replace(/^[/\\]+/, "");
  if (!id || !newEntryKind.value || !name) return;
  const path = selectedDirectory.value ? `${selectedDirectory.value}/${name}` : name;
  const result = await ws.perform(() => api.createProjectEntry(id, newEntryKind.value!, path));
  if (result) {
    newEntryKind.value = null;
    newEntryPath.value = "";
    await ws.refresh();
  }
}
const file = ref<{ path: string; changed: boolean; before: string | null; after: string; diff: string } | null>(null);
const newPath = ref(""), newProvider = ref<Provider>("deepseek"), newModel = ref("deepseek-v4-flash"), newPermission = ref("important"), newVerification = ref("auto"), newTest = ref(""), newError = ref("");
const settingsProvider = ref<Provider>("deepseek"), settingsEffort = ref("high"), settingsStyle = ref("standard"), settingsPermission = ref("important"), settingsVerification = ref("auto"), settingsTest = ref("");
const modelOptions = ref<Record<Provider, { selected: string; choices: string[] }> | null>(null);
const defaultModels: Record<Provider, string> = { deepseek: "deepseek-v4-flash", openai: "gpt-5.6-terra", openai_official: "gpt-6-astra" };
const newModelChoices = computed(() => modelOptions.value?.[newProvider.value]?.choices || []);
const effortIndex = ref(0);
const settingOptions = computed(() => efforts[settingsProvider.value]);
const composerEfforts = computed(() => efforts[ws.active.value?.provider || "deepseek"]);
const activityNames: Record<string, string> = { created: "会话已创建", user_message: "收到任务", assistant_message: "完整回答", workspace_entry_created: "项目条目已创建", task_contract: "建立任务契约", intent_clarification: "澄清任务意图", intent_classifier_fallback: "意图分类安全降级", context_compaction_fallback: "上下文摘要恢复", context_compressed: "上下文已压缩", context_trimmed: "上下文已裁剪", steer_queued: "运行中纠正已排队", steer_applied: "已切换任务目标", plan: "制定计划", plan_resumed: "恢复执行计划", context_prepared: "上下文已准备", model_waiting: "等待模型", model_retrying: "精简重试", model_response: "模型响应", model_diagnostic: "调用诊断", duplicate_corrected: "自动纠正重复动作", decision: "Agent 决策", observation: "工具结果", patch: "补丁已应用", permission_requested: "请求权限", permission_revised: "调整执行方案", permission_approved: "权限已允许", permission_execution_failed: "已允许，执行失败", permission_recovered: "权限状态已恢复", permission_denied: "权限被拒绝", interrupted: "执行已中断", paused: "任务已暂停", failed: "任务停止", completed: "任务完成", settings_updated: "配置已更新" };
const activityEvents = computed(() => ws.events.value.filter(e => showAllEvents.value || !["created", "context_prepared", "model_waiting"].includes(e.event_type)).slice().reverse());
const latestEvent = computed(() => activityEvents.value[0]);
activityNames.tool_call_started = "执行工具";
activityNames.tool_call_state = "工具调用状态";
Object.assign(activityNames, { skills_loaded: "指定技能已加载", skills_updated: "技能模式已保存", skill_imported: "技能已导入", skill_updated: "技能已更新", skill_deleted: "技能已移除", skill_permissions_repaired: "技能执行权限已修复" });
const workingPlan = computed(() => ws.active.value?.plan || []);
const completedPlan = computed(() => workingPlan.value.filter(item => item.status === "completed").length);
const compressionCount = computed(() => ws.events.value.filter(item => ["context_compressed", "context_trimmed"].includes(item.event_type)).length);
const workingDetail = computed(() => latestEvent.value && typeof latestEvent.value.payload.summary === "string" ? latestEvent.value.payload.summary : "正在准备下一步操作。");
const contextSummary = computed(() => { const item = [...ws.events.value].reverse().find(event => ["context_compressed", "context_trimmed"].includes(event.event_type)); if (!item) return "未触发"; const before = Number(item.payload.before_tokens || 0), after = Number(item.payload.after_tokens || 0); return before && after ? `${before.toLocaleString()} → ${after.toLocaleString()}` : "已完成"; });
function contextMetrics(events: Event[], estimated: number, actual: number | null | undefined, currentLimit?: number, running = false) {
  const modelChange = events.slice().reverse().find((event) => event.event_type === "settings_updated" && event.payload.model_changed === true)?.sequence || 0;
  const prepared = events.slice().reverse().find((event) => event.sequence > modelChange && ["context_prepared", "context_compressed", "context_trimmed"].includes(event.event_type));
  const compressions = events.filter((event) => ["context_compressed", "context_trimmed"].includes(event.event_type));
  const latest = compressions.at(-1);
  const previousResponse = events.slice().reverse().find((event) => event.sequence > modelChange && event.event_type === "model_response");
  const lastEstimate = estimated || Number(prepared?.payload.estimated_tokens || prepared?.payload.after_tokens || 0);
  const lastActual = actual || (!estimated && !running ? Number(previousResponse?.payload.input_tokens || 0) : 0);
  const measured = !running && lastActual > 0;
  const used = Math.max(0, measured ? lastActual : lastEstimate);
  const limit = Math.max(1, Number((estimated || actual) && currentLimit ? currentLimit : prepared?.payload.limit_tokens || currentLimit || 16_000));
  const before = Number(latest?.payload.before_tokens || 0);
  const after = Number(latest?.payload.after_tokens || 0);
  const latestCompression = latest ? `${latest.event_type === "context_compressed" ? "压缩" : "裁剪"}${before && after ? ` ${before.toLocaleString()} → ${after.toLocaleString()}` : "已完成"}` : "";
  const percent = used ? Math.min(100, (used / limit) * 100) : null;
  const remaining = percent === null ? "—" : `剩余 ${Math.max(0, 100 - percent).toFixed(percent < 1 ? 2 : 1)}%`;
  const detail = used ? `${measured ? "实际输入" : "预估输入"} ${used.toLocaleString()} / ${limit.toLocaleString()} Token` : `尚无模型调用 · 上限 ${limit.toLocaleString()} Token`;
  const actualDetail = measured ? "服务商返回实际输入；压缩按本地估算触发" : running ? "实际用量待返回；压缩按本地估算触发" : "实际用量未提供；压缩按本地估算触发";
  return { percent, remaining, detail, actualDetail, compressions: compressions.length, latestCompression };
}
const contextUsage = computed(() => contextMetrics(ws.events.value, Number(ws.active.value?.context_estimated_tokens || 0), ws.active.value?.context_actual_input_tokens, ws.active.value?.context_limit_tokens, ws.active.value?.status === "running"));

async function openFile(path: string) {
  if (!ws.activeId.value) return;
  const result = await ws.perform(() => api.fileChange(ws.activeId.value!, path));
  if (result) { file.value = result; fileView.value = result.changed ? "diff" : "after"; modal.value = "file"; }
}
function newConversation() { newPath.value = ""; newTest.value = ""; newError.value = ""; modal.value = "new"; }
async function createSession() {
  if (!newPath.value.trim()) { newError.value = "请输入项目目录的绝对路径"; return; }
  const result = await ws.perform(() => api.createSession({ repo_root: newPath.value.trim(), provider: newProvider.value, model: newModel.value, reasoning_effort: newProvider.value === "deepseek" ? "high" : "medium", permission_mode: newPermission.value, verification_mode: newVerification.value, test_command: newTest.value.trim() }));
  if (result) { modal.value = null; await ws.loadSessions(); ws.activeId.value = result.session_id; }
}
function preferredModel(provider: Provider, current?: string) {
  const choices = modelOptions.value?.[provider]?.choices || [];
  if (current && choices.includes(current)) return current;
  return modelOptions.value?.[provider]?.selected || choices[0] || defaultModels[provider];
}
function changeNewProvider() {
  newModel.value = preferredModel(newProvider.value);
}
function openSettings() {
  const s = ws.active.value;
  if (!s) return;
  settingsProvider.value = s.provider; settingsEffort.value = s.reasoning_effort; settingsStyle.value = s.response_style; settingsPermission.value = s.permission_mode; settingsVerification.value = s.verification_mode; settingsTest.value = s.test_command.join(" "); modal.value = "settings";
}
async function saveSettings() {
  await ws.settings({ provider: settingsProvider.value, model: preferredModel(settingsProvider.value, ws.active.value?.model), reasoning_effort: settingsEffort.value, response_style: settingsStyle.value, permission_mode: settingsPermission.value, verification_mode: settingsVerification.value, test_command: settingsTest.value });
  if (!ws.error.value) modal.value = null;
  else ws.error.value = `保存失败：${ws.error.value}`;
}
function statusLabel(s: Session) { return phases[s.activity || s.status] || s.status; }
async function saveComposerSettings(body: object) {
  if (!ws.active.value || ws.active.value.status === "running") return;
  await ws.settings(body);
}
async function changeComposerProvider(provider: Provider) {
  await saveComposerSettings({ provider, model: preferredModel(provider) });
}
function syncEffort() {
  const current = ws.active.value;
  if (!current) { effortIndex.value = 0; return; }
  const index = composerEfforts.value.findIndex(item => item[0] === current.reasoning_effort);
  effortIndex.value = Math.max(0, index < 0 ? 0 : index);
}
watch(() => [ws.active.value?.session_id, ws.active.value?.provider, ws.active.value?.reasoning_effort], syncEffort);
watch(newProvider, changeNewProvider);
watch(settingsProvider, (provider) => {
  const available = efforts[provider];
  if (!available.some(([value]) => value === settingsEffort.value)) {
    settingsEffort.value = available.some(([value]) => value === "high") ? "high" : available[0][0];
  }
});
onMounted(async () => { try { modelOptions.value = await api.models(); } catch { modelOptions.value = null; } });
</script>

<template>
  <div class="app-shell">
    <header class="titlebar"><div class="titlebar-drag pywebview-drag-region" @dblclick="windowAction('maximize')"><img :src="'/studio-brand'" alt="RAgent" /></div><div class="window-actions"><button title="最小化" @click="windowAction('minimize')">−</button><button title="最大化" @click="windowAction('maximize')">□</button><button title="关闭" @click="windowAction('close')">×</button></div></header>
    <div class="workspace" :class="{ 'with-sidebar': showSidebar, 'with-trace': showTrace }" :style="workspaceStyle">
      <aside class="sidebar">
        <div class="side-actions"><button class="primary" @click="newConversation">＋ 新建对话</button><button class="icon" :disabled="!ws.active.value" title="Git 工作区" @click="modal = 'git'">Git</button><button class="icon" title="升级验收" @click="modal = 'acceptance'">验收</button><button class="icon" title="API 与服务设置" @click="modal = 'api'">设置</button></div>
        <div class="section-label">工作区会话</div>
        <div class="session-list"><div v-for="item in ws.sessions.value" :key="item.session_id" class="session-row" :class="{ active: item.session_id === ws.activeId.value }"><button @click="ws.activeId.value = item.session_id"><strong>{{ item.title || '未命名任务' }}</strong><small>{{ statusLabel(item) }} · {{ item.step }} 步</small></button><button class="delete" title="删除会话" @click="ws.remove(item.session_id)">×</button></div><p v-if="!ws.sessions.value.length" class="git-empty">暂无会话</p></div>
        <section class="project-panel"><header><div><span>项目文件</span><strong>{{ ws.active.value?.repo_root || '请选择项目' }}</strong></div><nav><button :disabled="!ws.active.value || ws.active.value.status === 'running'" @click="beginEntry('file')">新建文件</button><button :disabled="!ws.active.value || ws.active.value.status === 'running'" @click="beginEntry('folder')">新建文件夹</button><b>{{ ws.files.value.length }}</b></nav></header><form v-if="newEntryKind" class="new-project-entry" @submit.prevent="createEntry"><div class="new-entry-location"><span>创建位置</span><strong>{{ selectedDirectory || '项目根目录' }}</strong><button v-if="selectedDirectory" type="button" @click="selectedDirectory = ''">改到根目录</button></div><input v-model="newEntryPath" autofocus :placeholder="newEntryKind === 'file' ? '输入文件名，例如 app.py' : '输入文件夹名称'" /><button class="primary" :disabled="!newEntryPath.trim()">创建</button><button type="button" @click="newEntryKind = null">×</button></form><div class="file-tree"><FileTree :files="ws.files.value" :directories="ws.directories.value" :changed="ws.changed.value" @open="openFile" @folder="selectedDirectory = $event" /></div></section>
      </aside>
      <div class="pane-resizer left" @pointerdown="resizePane('left', $event)" />
      <main class="conversation">
        <header class="hero"><button aria-label="切换侧栏" @click="showSidebar = !showSidebar">☰</button><div class="conversation-heading"><h1>{{ ws.active.value?.title || '开始一个编码任务' }}</h1><p>{{ ws.active.value ? `${ws.active.value.provider} · ${ws.active.value.model} · ${ws.active.value.verification_mode}` : '选择项目，然后用自然语言要求 Agent 阅读、修改并验证代码。' }}</p></div><div class="conversation-actions"><button v-if="ws.active.value && ['running','paused'].includes(ws.active.value.status)" :disabled="ws.busy.value || ws.pausePending.value" @click="ws.control">{{ ws.pausePending.value ? '正在暂停…' : ws.active.value.status === 'running' ? '暂停' : '继续' }}</button><button :disabled="!ws.active.value" @click="modal = 'skills'">技能</button><button :disabled="!ws.active.value" @click="openSettings">会话设置</button></div><button @click="showTrace = !showTrace">执行记录</button></header>
        <section ref="messages" class="messages" @scroll="ws.trackScroll"><article v-for="(message, index) in ws.visibleMessages.value" :key="`${index}-${message.role}`" :class="message.role"><img v-if="message.role === 'assistant'" class="assistant-avatar" :src="'/studio-brand'" alt="" /><MessageContent :content="message.content" /></article><div v-if="!ws.visibleMessages.value.length" class="welcome"><img :src="'/studio-brand'" alt="" /><h2>从项目开始</h2><p>让 RAgent 阅读代码、修改文件，或检查测试结果。</p></div><article v-if="ws.active.value?.status === 'running'" class="agent-working" aria-live="polite"><header class="working-header"><div><strong>{{ statusLabel(ws.active.value) }}</strong><small>受控执行中</small></div><b>进行中</b></header><div class="working-overview"><div class="working-current"><span class="working-live-dot" /><div><strong>{{ workingDetail }}</strong><small v-if="workingPlan.find(item => item.status === 'in_progress')">当前步骤 · {{ workingPlan.find(item => item.status === 'in_progress')?.title }}</small></div></div><div class="working-stats"><div><span>已完成步骤</span><strong>{{ completedPlan }}</strong></div><div><span>剩余步骤</span><strong>{{ Math.max(0, workingPlan.length - completedPlan) }}</strong></div></div></div><details v-if="workingPlan.length" class="working-details"><summary>查看执行步骤 <span>{{ completedPlan }} / {{ workingPlan.length }}</span></summary><ol class="working-plan"><li v-for="item in workingPlan" :key="item.key" :class="item.status"><span><b>{{ item.title }}</b><small v-if="item.note">{{ item.note }}</small></span></li></ol></details></article><article v-if="ws.active.value?.status === 'failed'" class="runtime-diagnostic" aria-live="polite"><header><div><strong>任务已停止</strong><small>代码与执行进度已经保存</small></div></header><p>{{ ws.active.value.failure_reason || '本轮执行未能完成，请查看右侧执行记录。' }}</p></article></section>
        <button v-if="!ws.followLatest.value" class="scroll-latest" @click="ws.latest">↓ 回到最新</button>
        <form class="composer" @submit.prevent="ws.send"><div class="compose-card"><textarea v-model="ws.draft.value" :disabled="!ws.active.value || ws.active.value.status === 'waiting_permission'" :placeholder="ws.active.value?.status === 'running' ? '输入纠正或追加要求，将在安全边界切换…' : '给 RAgent 一个任务…'" @keydown.enter.exact.prevent="ws.send" /><button type="submit" :disabled="!ws.active.value || !ws.draft.value.trim() || ws.sending.value">{{ ws.sending.value ? '…' : '➤' }}</button><footer class="model-controls"><template v-if="ws.active.value"><label>API<select :value="ws.active.value.provider" :disabled="ws.active.value.status === 'running' || ws.busy.value" @change="changeComposerProvider(($event.target as HTMLSelectElement).value as Provider)"><option value="deepseek">DeepSeek</option><option value="openai">OpenAI 中转站</option><option value="openai_official">OpenAI 官方</option></select></label><label>模型<select :value="ws.active.value.model" :disabled="ws.active.value.status === 'running' || ws.busy.value" @change="saveComposerSettings({ model: ($event.target as HTMLSelectElement).value })"><option v-for="item in modelOptions?.[ws.active.value.provider]?.choices || [ws.active.value.model]" :key="item">{{ item }}</option></select></label><label class="effort-control"><span>推理强度</span><input v-model.number="effortIndex" type="range" min="0" :max="Math.max(0, composerEfforts.length - 1)" :disabled="ws.active.value.status === 'running' || ws.busy.value" @change="saveComposerSettings({ reasoning_effort: composerEfforts[effortIndex]?.[0] })" /><span class="effort-level">{{ composerEfforts[effortIndex]?.[1] || '' }}</span></label></template><span v-else>未选择会话</span></footer></div></form>
      </main>
      <div class="pane-resizer right" @pointerdown="resizePane('right', $event)" />
      <aside class="trace"><header><div class="trace-heading"><strong>执行记录</strong><small>{{ activityEvents.length }} 条记录</small><button class="trace-detail-toggle" @click="showAllEvents = !showAllEvents">{{ showAllEvents ? '精简' : '全部记录' }}</button></div><span class="state" :class="ws.active.value?.status || 'idle'"><i />{{ statusLabel(ws.active.value || ({ status: 'idle' } as Session)) }}</span></header><div class="activity-feed"> <details v-for="event in activityEvents" :key="event.sequence" class="activity-entry" :open="['failed','paused'].includes(event.event_type)"><summary><span class="activity-head"><strong>{{ activityNames[event.event_type] || event.event_type }}</strong><time>{{ new Date(event.created_at).toLocaleTimeString() }}</time></span><small>{{ typeof event.payload.summary === 'string' ? event.payload.summary : '查看详情' }}</small></summary><pre>{{ JSON.stringify(event.payload, null, 2) }}</pre></details></div><footer class="trace-summary"><div><span>累计决策步</span><strong>{{ ws.active.value?.step || 0 }}</strong></div><div><span>修改文件</span><strong>{{ ws.active.value?.changed_files.length || 0 }}</strong></div><div class="context-usage"><span>最近一次模型上下文</span><strong>{{ contextUsage.remaining }}</strong><small>{{ contextUsage.detail }}<br />{{ contextUsage.actualDetail }}</small><i><b :style="{ width: `${Math.min(contextUsage.percent || 0, 100)}%` }" /></i></div><div class="context-compression"><span>上下文处理</span><strong>{{ contextUsage.compressions ? `${contextUsage.compressions} 次` : '未触发' }}</strong><small v-if="contextUsage.latestCompression">{{ contextUsage.latestCompression }}</small></div><div><span>当前阶段</span><strong :class="ws.active.value?.status || 'idle'">{{ statusLabel(ws.active.value || ({ status: 'idle' } as Session)) }}</strong></div></footer></aside>
    </div>
    <div v-if="ws.error.value" class="toast" role="alert" @click="ws.error.value = ''">{{ ws.error.value }}</div>
    <PermissionDialog v-if="ws.active.value?.pending_permission" :permission="ws.active.value.pending_permission" :busy="ws.busy.value" @decide="ws.permission" />
    <Modal v-if="modal === 'api'" title="设置" @close="modal = null"><p>选择要管理的项目配置。</p><div class="form-actions"><button @click="modal = 'api-config'">API 与模型配置</button><button @click="modal = 'mcp'">MCP 服务管理</button><button @click="modal = 'sandbox'">执行沙箱</button></div></Modal>
    <SandboxSettings v-if="modal === 'sandbox'" @close="modal = null" />
    <ApiSettings v-if="modal === 'api-config'" @close="modal = null" />
    <Skills v-if="modal === 'skills' && ws.active.value" :id="ws.active.value.session_id" @close="modal = null" />
    <McpSettings v-if="modal === 'mcp'" :repo-root="ws.active.value?.repo_root || ''" @close="modal = null" />
    <GitPanel v-if="modal === 'git' && ws.active.value" :id="ws.active.value.session_id" @close="modal = null" />
    <UpgradeAcceptance v-if="modal === 'acceptance'" @close="modal = null" @back="modal = 'api'" />
    <Modal v-if="modal === 'new'" title="新建对话" @close="modal = null"><p>先选择项目目录；创建后可在会话设置中调整模型和权限。</p><label>项目文件夹<div class="path-row"><input v-model="newPath" required placeholder="C:\项目\路径" /><button type="button" @click="modal = 'folders'">选择</button></div></label><label>模型<select v-model="newProvider" @change="changeNewProvider"><option v-for="item in providers" :key="item[0]" :value="item[0]">{{ item[1] }}</option></select></label><label>模型名称<select v-if="newModelChoices.length" v-model="newModel"><option v-for="item in newModelChoices" :key="item">{{ item }}</option></select><input v-else v-model="newModel" placeholder="模型名称" /></label><label>权限级别<select v-model="newPermission"><option value="ask">逐项批准</option><option value="important">仅重要操作批准</option><option value="full">完全批准</option></select></label><label>验证模式<select v-model="newVerification"><option value="auto">自动验证</option><option value="quick">快速模式</option><option value="strict">严格验证</option></select></label><label>验证命令（可选）<input v-model="newTest" placeholder="留空按项目自动选择；严格模式需填写" /></label><p v-if="newError" class="form-error">{{ newError }}</p><div class="form-actions"><button @click="modal = null">取消</button><button class="primary" @click="createSession">创建对话</button></div></Modal>
    <FolderPicker v-if="modal === 'folders'" @close="modal = 'new'" @select="(path) => { newPath = path; modal = 'new'; }" />
    <Modal v-if="modal === 'settings' && ws.active.value" title="会话设置" :busy="ws.busy.value" @close="modal = null"><label>服务商<select v-model="settingsProvider"><option v-for="item in providers" :key="item[0]" :value="item[0]">{{ item[1] }}</option></select></label><label>推理强度<select v-model="settingsEffort"><option v-for="item in settingOptions" :key="item[0]" :value="item[0]">{{ item[1] }}</option></select></label><label>回答风格<select v-model="settingsStyle"><option value="concise">精简</option><option value="standard">标准</option><option value="teaching">教学</option></select></label><label>权限级别<select v-model="settingsPermission"><option value="ask">逐项批准</option><option value="important">仅重要操作批准</option><option value="full">完全批准</option></select></label><label>验证模式<select v-model="settingsVerification"><option value="auto">自动验证</option><option value="quick">快速模式</option><option value="strict">严格验证</option></select></label><label>验证命令（可选）<input v-model="settingsTest" placeholder="留空按项目选择" /></label><div class="form-actions"><button @click="modal = null">取消</button><button class="primary" :disabled="ws.busy.value" @click="saveSettings">{{ ws.busy.value ? '保存中…' : '保存' }}</button></div></Modal>
    <Modal v-if="modal === 'file' && file" :title="file.path" :wide="true" @close="modal = null"><p>{{ file.changed ? '已修改 · 绿色为新增，红色为删除' : '未修改' }}</p><div class="form-actions"><button :class="{ active: fileView === 'before' }" @click="fileView = 'before'">更改前</button><button :class="{ active: fileView === 'after' }" @click="fileView = 'after'">更改后</button><button v-if="file.diff" :class="{ active: fileView === 'diff' }" @click="fileView = 'diff'">差异对比</button></div><pre class="vue-code">{{ fileView === 'before' ? (file.before || '没有修改前快照') : fileView === 'diff' ? file.diff : file.after }}</pre></Modal>
  </div>
</template>

<script lang="ts">
declare global {
  interface Window {
    pywebview?: { api?: { minimize?: () => void; maximize?: () => void; close?: () => void; sandbox_token?: () => Promise<string> } };
  }
}
export {};
</script>
