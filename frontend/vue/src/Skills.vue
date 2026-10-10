<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import Modal from "./Modal.vue";

type Mode = "auto" | "pinned" | "disabled";
type Entry = { path: string; size: number };
type Skill = { name: string; description: string; compatibility: string; files: Entry[]; version: string; content?: string; base_directory: string };
type ReleaseVersion = { created_at: number; checks: { static: string; smoke_cases: number; executable_smoke: string }; references: number; metrics: { executions: number; errors: number; mean_latency_ms: number; task_samples: number; task_successes: number } };
type Release = { active: string | null; candidate: string | null; percent: number; deleted?: boolean; notice: string; watch_error?: string; versions: Record<string, ReleaseVersion>; policy: { auto_rollback: boolean; min_samples: number; failure_rate: number; max_latency_ms: number } };
type Listing = { items: Skill[]; modes: Record<string, Mode>; diagnostics: { name: string; message: string; version?: string }[]; active: string[]; locked: boolean; releases: Record<string, Release>; bindings: Record<string, string> };
type ImportBody = { content?: string; files?: { path: string; data: string }[]; archive?: string };
type Preview = Skill & { existing_version: string | null; total_bytes: number };
type Resource = { path: string; kind: string; content: string; size: number; truncated?: boolean };
const props = defineProps<{ id: string }>();
const emit = defineEmits<{ close: [] }>();
const items = ref<Skill[]>([]), diagnostics = ref<Listing["diagnostics"]>([]);
const modes = ref<Record<string, Mode>>({}), saved = ref<Record<string, Mode>>({});
const active = ref<string[]>([]), locked = ref(false), busy = ref(false);
const error = ref(""), notice = ref(""), query = ref("");
const selected = ref<Skill | null>(null), resource = ref<Resource | null>(null);
const view = ref<"detail" | "import" | "history">("detail"), editing = ref(false), editText = ref("");
const releases = ref<Record<string, Release>>({}), bindings = ref<Record<string, string>>({});
const releasePercent = ref(100), autoRollback = ref(false), minSamples = ref(5), failureRate = ref(0.5), maxLatency = ref(0);
const smokeConfirm = ref(false), smokeResult = ref("");
const restoring = ref<{ name: string; version: string } | null>(null);
const currentRelease = computed(() => selected.value ? releases.value[selected.value.name] : undefined);
let refreshTimer: ReturnType<typeof setInterval> | undefined;
const source = ref<"file" | "folder" | "zip" | "text">("folder"), text = ref("");
const upload = ref<ImportBody | null>(null), uploadLabel = ref("");
const preview = ref<Preview | null>(null), trusted = ref(false), replaceConfirmed = ref(false);
const deleting = ref(false), repairing = ref(false), closing = ref(false);
const modeLabel: Record<Mode, string> = { auto: "自动选择", pinned: "固定启用", disabled: "禁用" };
const signature = (value: Record<string, Mode>) => JSON.stringify(Object.entries(value).sort());
const dirty = computed(() => signature(modes.value) !== signature(saved.value));
const editDirty = computed(() => editing.value && editText.value !== selected.value?.content);
const filtered = computed(() => items.value.filter(i => `${i.name} ${i.description}`.toLowerCase().includes(query.value.toLowerCase())));
const readonly = computed(() => busy.value || locked.value);
const base = computed(() => `/studio-api/sessions/${encodeURIComponent(props.id)}/skills`);
const size = (n: number) => n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;

async function request<T>(suffix = "", method = "GET", body?: object): Promise<T> {
  const response = await fetch(base.value + suffix, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : Array.isArray(data.detail) ? data.detail.map((d: { msg: string }) => d.msg).join("；") : "技能请求失败");
  return data as T;
}
async function perform(action: () => Promise<void>) {
  busy.value = true; error.value = ""; closing.value = false;
  try { await action(); } catch (e) { error.value = (e as Error).message; } finally { busy.value = false; }
}
async function load(preserve = false) {
  const data = await request<Listing>();
  items.value = data.items; diagnostics.value = data.diagnostics; active.value = data.active; locked.value = data.locked;
  releases.value = data.releases ?? {}; bindings.value = data.bindings ?? {};
  modes.value = preserve ? Object.fromEntries(data.items.map(i => [i.name, modes.value[i.name] ?? data.modes[i.name] ?? "disabled"])) : { ...data.modes };
  saved.value = { ...data.modes };
}
function mayLeaveEditor() {
  if (editDirty.value) { error.value = "说明有未保存修改，请先保存或取消编辑。"; return false; }
  return true;
}
async function choose(item: Skill) {
  if (!mayLeaveEditor()) return;
  selected.value = await request<Skill>(`/${encodeURIComponent(item.name)}`);
  editing.value = false; resource.value = null; deleting.value = false; repairing.value = false; view.value = "detail";
  const release = releases.value[item.name];
  releasePercent.value = release?.candidate ? release.percent : 100;
  autoRollback.value = release?.policy.auto_rollback ?? false;
  minSamples.value = release?.policy.min_samples ?? 5;
  failureRate.value = release?.policy.failure_rate ?? 0.5;
  maxLatency.value = release?.policy.max_latency_ms ?? 0;
  smokeConfirm.value = false; smokeResult.value = "";
}
function openImport() {
  if (!mayLeaveEditor()) return;
  view.value = "import"; deleting.value = false; error.value = "";
}
function openHistory() {
  if (mayLeaveEditor()) view.value = "history";
}
function resetPreview() { preview.value = null; trusted.value = false; replaceConfirmed.value = false; }
function changeSource(value: typeof source.value) {
  source.value = value; upload.value = null; uploadLabel.value = ""; resetPreview();
}
async function encode(file: File) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 8192) binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
  return btoa(binary);
}
async function pick(event: Event) {
  const input = event.target as HTMLInputElement;
  const files = Array.from(input.files ?? []); input.value = "";
  if (!files.length) return;
  await perform(async () => {
    resetPreview(); upload.value = null; uploadLabel.value = "";
    if (files.length > 200 || files.reduce((sum, f) => sum + f.size, 0) > 8 * 1024 * 1024) throw new Error("最多 200 个文件，合计不能超过 8 MB。");
    if (source.value === "file") {
      const content = await files[0]!.text();
      if (content.length > 12000) throw new Error("SKILL.md 不能超过 12000 字符。");
      upload.value = { content };
    } else if (source.value === "zip") upload.value = { archive: await encode(files[0]!) };
    else {
      if (files.some(f => f.size > 2 * 1024 * 1024)) throw new Error("单个文件不能超过 2 MB。");
      upload.value = { files: await Promise.all(files.map(async f => ({ path: f.webkitRelativePath || f.name, data: await encode(f) }))) };
    }
    uploadLabel.value = source.value === "folder" ? `${files[0]!.webkitRelativePath.split('/')[0]} · ${files.length} 个文件` : files[0]!.name;
  });
}
async function inspectImport() {
  resetPreview();
  const body = source.value === "text" ? { content: text.value } : upload.value;
  if (!body) throw new Error("请先选择文件或文件夹。");
  preview.value = await request<Preview>("/preview", "POST", body);
}
async function importSkill() {
  if (!preview.value || !trusted.value || (preview.value.existing_version && !replaceConfirmed.value)) return;
  const result = await request<Skill & { backup: string | null }>("", "POST", {
    ...(source.value === "text" ? { content: text.value } : upload.value),
    replace: !!preview.value.existing_version, expected_version: preview.value.existing_version,
  });
  notice.value = result.backup ? `已更新 ${result.name}。原版本备份：${result.backup}` : `已导入 ${result.name}，默认允许自动选择。`;
  resetPreview(); upload.value = null; uploadLabel.value = ""; text.value = "";
  await load(true); await choose(result);
}
async function saveModes() {
  const result = await request<{ modes: Record<string, Mode> }>("", "PUT", { modes: modes.value });
  saved.value = { ...result.modes }; modes.value = { ...result.modes }; active.value = [];
  notice.value = "本会话的技能使用方式已保存。";
}
async function saveEdit() {
  if (!selected.value) return;
  const result = await request<Skill & { backup: string }>(`/${encodeURIComponent(selected.value.name)}`, "PATCH", { content: editText.value, expected_version: selected.value.version });
  selected.value = result; editing.value = false;
  notice.value = `说明已更新，配套资源保留。原版本备份：${result.backup}`;
  await load(true);
}
async function removeSkill() {
  if (!selected.value) return;
  const result = await request<{ backup: string }>(`/${encodeURIComponent(selected.value.name)}?version=${encodeURIComponent(selected.value.version)}`, "DELETE");
  notice.value = `已从项目技能中移除，文件可从备份恢复：${result.backup}`;
  selected.value = null; resource.value = null; deleting.value = false; editing.value = false;
  await load(true);
}
async function repairPermissions() {
  if (!selected.value) return;
  const result = await request<Skill & { backup: string }>(`/${encodeURIComponent(selected.value.name)}/repair?version=${encodeURIComponent(selected.value.version)}`, "POST");
  selected.value = result; repairing.value = false;
  notice.value = `已按项目权限重新保存，脚本和资料内容不变。原包备份：${result.backup}。执行仍受审批和沙箱控制。`;
  await load(true);
}
async function showResource(path: string) {
  if (!selected.value) return;
  resource.value = await request<Resource>(`/${encodeURIComponent(selected.value.name)}/resource?path=${encodeURIComponent(path)}`);
}
async function publishRelease() {
  if (!selected.value) return;
  const data = await request<{ notice: string }>(`/${encodeURIComponent(selected.value.name)}/release`, "PUT", {
    version: selected.value.version, percent: releasePercent.value, auto_rollback: autoRollback.value,
    min_samples: minSamples.value, failure_rate: failureRate.value, max_latency_ms: maxLatency.value,
  });
  notice.value = data.notice + "；只影响后续新任务。";
  await load(true);
}
async function runSmoke() {
  if (!selected.value || !smokeConfirm.value) return;
  const data = await request<{ passed: boolean; cases: object[]; notice: string }>(`/${encodeURIComponent(selected.value.name)}/smoke`, "POST", { version: selected.value.version, confirmed: true });
  smokeResult.value = JSON.stringify(data, null, 2); smokeConfirm.value = false;
  notice.value = data.passed ? "冒烟测试通过，可以发布该版本。" : "冒烟测试未通过，新版保持未发布。";
  await load(true);
}
async function restoreVersion() {
  if (!restoring.value) return;
  const { name, version } = restoring.value;
  const installed = items.value.find(i => i.name === name);
  const expectedVersion = installed?.version ?? diagnostics.value.find(d => d.name === name)?.version ?? null;
  const result = await request<Skill>(`/${encodeURIComponent(name)}/restore`, "POST", { version, expected_version: expectedVersion });
  restoring.value = null; notice.value = "历史版本已恢复；已有任务不切换，外部操作未撤销。";
  await load(true); await choose(result);
}
function close() {
  if (busy.value) return;
  if (dirty.value || editDirty.value) { closing.value = true; return; }
  emit("close");
}
onMounted(() => {
  void perform(load);
  refreshTimer = setInterval(() => {
    if (!busy.value) void load(true).catch(e => { error.value = (e as Error).message; });
  }, 3000);
});
onUnmounted(() => { if (refreshTimer) clearInterval(refreshTimer); });
</script>

<template>
  <Modal title="技能管理" :busy="busy" :wide="true" @close="close">
    <div class="skills-intro"><p>完整技能包 · 自动选择 · 按需加载</p><span>{{ items.length }} 个技能</span></div>
    <p v-if="locked" class="vue-notice">任务正在运行或等待审批，暂时只能查看。结束后点刷新即可管理。</p>
    <div v-if="closing" class="skills-confirm" role="alert"><p>有未保存的修改，确定放弃并关闭？</p><button @click="closing = false">继续编辑</button><button @click="emit('close')">放弃修改并关闭</button></div>
    <p v-if="error" class="form-error" role="alert">{{ error }}</p>
    <p v-if="notice" class="skills-notice" role="status">{{ notice }}</p>
    <div class="skills-layout">
      <aside class="skills-sidebar">
        <div class="skills-toolbar"><button :disabled="readonly" @click="openImport">＋ 导入技能</button><button :disabled="busy" @click="perform(async () => { if (!mayLeaveEditor()) return; await load(true); if (selected && items.some(i => i.name === selected?.name)) await choose(selected); })">刷新</button><button :disabled="busy" @click="openHistory">历史版本</button></div>
        <input v-model="query" aria-label="搜索技能" placeholder="搜索名称或描述" />
        <div class="skills-list">
          <p v-if="!filtered.length" class="skills-muted">{{ items.length ? '没有匹配技能' : '暂无技能，导入文件夹、ZIP 或 SKILL.md 开始使用。' }}</p>
          <button v-for="item in filtered" :key="item.name" class="skills-item" :class="{ chosen: selected?.name === item.name && view === 'detail' }" :disabled="busy" @click="perform(() => choose(item))"><strong>{{ item.name }}</strong><span class="skills-description">{{ item.description }}</span><small>{{ modeLabel[modes[item.name] ?? 'disabled'] }}<span v-if="active.includes(item.name)"> · 本次已加载</span></small></button>
        </div>
        <details v-if="diagnostics.length" class="skills-diagnostics"><summary>{{ diagnostics.length }} 个技能无法加载</summary><p v-for="d in diagnostics" :key="d.name"><strong>{{ d.name }}</strong><br />{{ d.message }}</p></details>
      </aside>
      <main class="skills-main">
        <template v-if="view === 'history'">
          <h3>历史版本与恢复</h3><p class="skills-muted">恢复只影响新任务，不撤销已执行的外部操作。版本保留供暂停任务恢复，不自动删除。</p>
          <div v-if="restoring" class="skills-confirm"><p>确认将 {{ restoring.name }} 恢复为 {{ restoring.version.slice(0, 12) }}？当前文件会备份。</p><button :disabled="readonly" @click="perform(restoreVersion)">确认恢复</button><button @click="restoring = null">取消</button></div>
          <section v-for="(release, name) in releases" :key="name" class="skills-preview">
            <h4>{{ name }} <small v-if="release.deleted">已删除，可恢复</small></h4>
            <p class="skills-muted">{{ release.notice }}</p>
            <ul class="skills-files"><li v-for="(record, version) in release.versions" :key="version"><span>{{ String(version).slice(0, 12) }} · {{ new Date(record.created_at * 1000).toLocaleString() }}<br />执行 {{ record.metrics.executions }} · 失败 {{ record.metrics.errors }} · 平均 {{ record.metrics.mean_latency_ms }} ms · 使用中 {{ record.references }}<br />静态 {{ record.checks.static }} · 冒烟 {{ record.checks.executable_smoke }}</span><button :disabled="readonly" @click="restoring = { name: String(name), version: String(version) }">恢复</button></li></ul>
          </section>
          <p v-if="!Object.keys(releases).length" class="skills-muted">升级后导入或加载技能时会建立历史版本。</p>
        </template>
        <template v-else-if="view === 'import'">
          <h3>导入技能</h3><p class="skills-muted">选择一个含 SKILL.md 的技能目录，脚本、资料和模板会一并保留。</p>
          <div class="skills-tabs"><button v-for="tab in (['folder', 'zip', 'file', 'text'] as const)" :key="tab" :class="{ active: source === tab }" :disabled="readonly" @click="changeSource(tab)">{{ { folder: '文件夹', zip: 'ZIP 包', file: 'SKILL.md', text: '粘贴文本' }[tab] }}</button></div>
          <label v-if="source === 'folder'">选择完整技能文件夹<input type="file" webkitdirectory multiple :disabled="readonly" @change="pick" /></label>
          <label v-else-if="source === 'zip'">选择 ZIP 文件<input type="file" accept=".zip" :disabled="readonly" @change="pick" /></label>
          <label v-else-if="source === 'file'">选择 SKILL.md<input type="file" accept=".md" :disabled="readonly" @change="pick" /></label>
          <label v-else>技能说明<textarea v-model="text" rows="9" maxlength="12000" :disabled="readonly" placeholder="---&#10;name: my-skill&#10;description: 适用场景&#10;---&#10;具体工作流程" @input="resetPreview" /></label>
          <p v-if="uploadLabel" class="skills-muted">{{ uploadLabel }}</p>
          <button :disabled="readonly || (source === 'text' ? !text.trim() : !upload)" @click="perform(inspectImport)">检查并预览</button>
          <section v-if="preview" class="skills-preview">
            <h4>{{ preview.name }} <small>{{ preview.existing_version ? '更新已有技能' : '新技能' }}</small></h4><p>{{ preview.description }}</p>
            <p v-if="preview.compatibility" class="skills-muted">环境要求：{{ preview.compatibility }}</p>
            <p class="skills-muted">{{ preview.files.length }} 个文件 · {{ size(preview.total_bytes) }}</p>
            <ul class="skills-files"><li v-for="file in preview.files" :key="file.path"><span>{{ file.path }}</span><small>{{ size(file.size) }}</small></li></ul>
            <label class="vue-check"><input v-model="trusted" type="checkbox" :disabled="readonly" />我已确认来源可信。导入不会执行脚本或安装依赖。</label>
            <label v-if="preview.existing_version" class="vue-check"><input v-model="replaceConfirmed" type="checkbox" :disabled="readonly" />确认替换整个同名技能包，旧版本将保留备份。</label>
            <button class="primary" :disabled="readonly || !trusted || (!!preview.existing_version && !replaceConfirmed)" @click="perform(importSkill)">{{ preview.existing_version ? '确认更新' : '确认导入' }}</button>
          </section>
        </template>
        <template v-else-if="selected">
          <div class="skills-detail-head"><h3>{{ selected.name }}</h3><span class="skills-muted">{{ selected.files.length }} 个文件</span></div><p>{{ selected.description }}</p>
          <p class="skills-muted">安装 {{ selected.version.slice(0, 12) }} · 发布 {{ currentRelease?.active?.slice(0, 12) ?? '待发布' }} · 本任务 {{ bindings[selected.name]?.slice(0, 12) ?? '未绑定' }}</p>
          <p v-if="items.find(i => i.name === selected?.name)?.version !== selected.version" class="vue-notice">文件已在外部变更，当前编辑内容未被覆盖；请刷新后再保存。</p>
          <p v-if="currentRelease?.watch_error" class="form-error">新版校验失败，保留上次发布版本：{{ currentRelease.watch_error }}</p>
          <p v-if="currentRelease" class="skills-muted">{{ currentRelease.notice }}</p>
          <label>本会话使用方式<select v-model="modes[selected.name]" :disabled="readonly"><option value="auto">自动选择</option><option value="pinned">固定启用</option><option value="disabled">禁用</option></select></label>
          <p class="skills-muted">{{ modes[selected.name] === 'auto' ? '只提供名称和描述，由模型按任务加载；也可用 $' + selected.name + ' 指定。' : modes[selected.name] === 'pinned' ? '本会话每次请求都加载完整说明，适合持续使用的工作流。' : '不向模型提供此技能，也不允许按需激活。' }}</p>
          <p v-if="selected.compatibility" class="skills-muted">环境要求：{{ selected.compatibility }}</p>
          <details class="skills-location"><summary>存放位置</summary><p>{{ selected.base_directory }}</p></details>
          <details class="skills-release"><summary>发布、灰度与冒烟测试</summary>
            <p class="skills-muted">任务绑定完整版本。灰度按新任务分流，不在任务中途切换。没有样例时只做静态检查。</p>
            <label>新版流量（%）<input v-model.number="releasePercent" type="number" min="0" max="100" :disabled="readonly" /></label>
            <label class="vue-check"><input v-model="autoRollback" type="checkbox" :disabled="readonly" />指标异常自动回退后续任务</label>
            <template v-if="autoRollback"><label>最少执行样本<input v-model.number="minSamples" type="number" min="3" max="100" :disabled="readonly" /></label><label>失败比例阈值（0–1）<input v-model.number="failureRate" type="number" min="0.01" max="1" step="0.1" :disabled="readonly" /></label><label>平均耗时阈值（毫秒，0 表示不检查）<input v-model.number="maxLatency" type="number" min="0" max="600000" :disabled="readonly" /></label></template>
            <button :disabled="readonly || editing" @click="perform(publishRelease)">保存发布策略</button>
            <template v-if="currentRelease?.versions[selected.version]?.checks.smoke_cases"><p class="skills-muted">执行 tests/smoke.json 中的脚本，仍可能修改工作区或产生允许的外部副作用。</p><label class="vue-check"><input v-model="smokeConfirm" type="checkbox" :disabled="readonly" />确认在默认沙箱中执行这些测试</label><button :disabled="readonly || !smokeConfirm" @click="perform(runSmoke)">运行冒烟测试</button></template>
            <pre v-if="smokeResult" class="vue-code">{{ smokeResult }}</pre>
          </details>
          <div class="skills-toolbar"><button v-if="!editing" :disabled="readonly" @click="editing = true; editText = selected.content ?? ''; resource = null">编辑说明</button><template v-else><button class="primary" :disabled="readonly || !editDirty" @click="perform(saveEdit)">保存说明</button><button :disabled="busy" @click="editing = false">取消编辑</button></template><button :disabled="readonly || editing" @click="repairing = !repairing; deleting = false">修复执行权限</button><button :disabled="readonly || editing" @click="deleting = !deleting; repairing = false">删除技能</button></div>
          <div v-if="repairing" class="skills-confirm"><p>适用于旧版导入后脚本报 Permission denied：将原文件备份，再按项目的权限继承重新保存。不会新增账户权限、关闭沙箱或运行脚本；使用方式保持不变。</p><button :disabled="readonly" @click="repairing = false">取消</button><button :disabled="readonly" @click="perform(repairPermissions)">确认修复并备份</button></div>
          <div v-if="deleting" class="skills-confirm"><p>从项目移除 {{ selected.name }}？后续新任务不再加载，已有任务仍使用绑定版本；原文件会移入备份目录。</p><button :disabled="readonly" @click="deleting = false">取消</button><button class="danger" :disabled="readonly" @click="perform(removeSkill)">确认移除并备份</button></div>
          <textarea v-if="editing" v-model="editText" rows="15" maxlength="12000" :disabled="readonly" aria-label="编辑技能说明" class="skills-editor" />
          <details v-else><summary>完整说明</summary><pre class="vue-code">{{ selected.content }}</pre></details>
          <h4>资源文件</h4><ul class="skills-files"><li v-for="file in selected.files" :key="file.path"><button :disabled="busy" @click="perform(() => showResource(file.path))">{{ file.path }}</button><small>{{ size(file.size) }}</small></li></ul>
          <section v-if="resource"><div class="skills-detail-head"><h4>{{ resource.path }}</h4><button @click="resource = null">收起</button></div><pre v-if="resource.kind === 'text'" class="vue-code">{{ resource.content }}</pre><p v-else class="skills-muted">二进制资源（{{ size(resource.size) }}），已保留在技能包中。</p><p v-if="resource.truncated" class="skills-muted">这里只预览前 32000 字符，原文件未被截断。</p></section>
        </template>
        <div v-else class="skills-empty"><h3>为任务添加可复用的工作流</h3><p>从左侧选择技能查看详情，或导入完整技能包。</p><button :disabled="readonly" @click="openImport">导入技能</button></div>
      </main>
    </div>
    <footer class="skills-footer"><span class="skills-muted">{{ dirty ? '使用方式有未保存修改' : '技能不会增加执行权限，脚本仍受审批和沙箱控制。' }}</span><div><button :disabled="busy" @click="close">关闭</button><button class="primary" :disabled="readonly || !dirty" @click="perform(saveModes)">保存使用方式</button></div></footer>
  </Modal>
</template>

<style scoped>
button{border:1px solid var(--line);border-radius:9px;background:#2c2d2f;color:inherit;padding:8px 12px;font:inherit;cursor:pointer;transition:border-color .15s,background .15s}button:hover:not(:disabled){border-color:#777;background:#353638}button:disabled{opacity:.45;cursor:default}button.primary{border-color:#5f9d86;background:rgba(103,207,166,.12);color:#b9e4d3}button.danger{border-color:#a96f6f;color:#edbaba}button:focus-visible{outline:2px solid #67cfa6;outline-offset:2px}
.skills-intro,.skills-toolbar,.skills-detail-head,.skills-footer,.skills-footer>div{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.skills-intro{justify-content:space-between;margin-bottom:16px}.skills-intro p{margin:0}.skills-intro span,.skills-muted{color:var(--muted,#a6abb3);font-size:13px;line-height:1.7}
.skills-layout{display:grid;grid-template-columns:240px minmax(0,1fr);gap:22px;min-height:380px}.skills-sidebar{min-width:0;border-right:1px solid var(--line);padding-right:18px}.skills-sidebar>input{margin:14px 0;width:100%}.skills-list{max-height:48vh;overflow:auto;display:grid;gap:8px;align-content:start}.skills-item{text-align:left!important;display:grid;gap:6px;width:100%;padding:13px!important;background:transparent!important;border:1px solid var(--line)!important;border-radius:10px!important;min-width:0}.skills-item.chosen{background:rgba(103,207,166,.08)!important;border-color:#67cfa6!important}.skills-item strong{overflow-wrap:anywhere}.skills-description{font-size:12px;line-height:1.6;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;color:var(--muted,#a6abb3)}.skills-item small{font-size:11px;color:#90b6a8}
.skills-main{min-width:0;max-height:60vh;overflow:auto;padding-right:4px}.skills-main h3{margin:0 0 12px}.skills-main h4{margin:18px 0 10px}.skills-main p{overflow-wrap:anywhere}.skills-main>label{margin:14px 0}.skills-main details{margin:16px 0}.skills-tabs{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0}.skills-tabs .active{border-color:#67cfa6;background:rgba(103,207,166,.08)}.skills-preview{border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:18px}.skills-preview h4{margin-top:0}.skills-preview h4 small{font-size:11px;font-weight:400;margin-left:8px;color:#90b6a8}.skills-preview .vue-check{font-size:13px;margin:14px 0;line-height:1.6}
.skills-files{list-style:none;padding:0;margin:10px 0;max-height:200px;overflow:auto;border:1px solid var(--line);border-radius:8px}.skills-files li{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:8px 10px;font-size:12px;border-bottom:1px solid var(--line)}.skills-files li:last-child{border-bottom:0}.skills-files span,.skills-files button{overflow-wrap:anywhere;min-width:0;text-align:left}.skills-files button{padding:0!important;border:0!important;background:none!important}.skills-files small{white-space:nowrap;color:var(--muted,#a6abb3)}.skills-location p{font:12px/1.7 Consolas,monospace;overflow-wrap:anywhere}.skills-editor{font:12px/1.7 Consolas,monospace;margin-top:14px}.skills-footer{justify-content:space-between;margin-top:20px;padding-top:16px;border-top:1px solid var(--line)}.skills-footer>span{flex:1;min-width:160px}.skills-footer>div{flex:none}.skills-notice{font-size:13px;line-height:1.7;color:#a4d6c1;background:rgba(103,207,166,.06);padding:10px 12px;border-radius:8px;overflow-wrap:anywhere}.skills-confirm{border:1px solid #a78658;border-radius:10px;padding:12px;margin:12px 0;font-size:13px}.skills-confirm p{margin-top:0}.skills-confirm button{margin-right:8px}.skills-diagnostics{font-size:12px;line-height:1.7;margin-top:16px;color:#e0b787}.skills-diagnostics p{overflow-wrap:anywhere}.skills-empty{padding:55px 12px;text-align:center;color:var(--muted,#a6abb3)}
@media(max-width:760px){.skills-layout{grid-template-columns:1fr;gap:16px}.skills-sidebar{border-right:0;border-bottom:1px solid var(--line);padding:0 0 16px}.skills-list{max-height:180px}.skills-main{max-height:none}.skills-footer{align-items:flex-start}.skills-footer>div{margin-left:auto}}
</style>
