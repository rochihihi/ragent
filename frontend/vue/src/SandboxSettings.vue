<script setup lang="ts">
import { onMounted, ref } from "vue";
import { api, type SandboxSettings } from "../../src/api";
import Modal from "./Modal.vue";

defineEmits<{ close: [] }>();
const policy = ref<SandboxSettings>({ mode: "required", allow_approved_host_execution: true, allowed_domains: [], tool_read_paths: [] });
const domains = ref(""), paths = ref(""), configPath = ref("");
const windows = ref(false), loaded = ref(false), busy = ref(false), error = ref(""), message = ref("");
const status = ref<{ ready: boolean; startupVerified?: boolean; errors: string[]; warnings: string[] } | null>(null);
async function perform(work: () => Promise<void>) {
  busy.value = true; error.value = ""; message.value = "";
  try { await work(); } catch (err) { error.value = err instanceof Error ? err.message : String(err); }
  finally { busy.value = false; }
}
onMounted(async () => { await perform(async () => {
  const result = await api.sandboxSettings();
  policy.value = result.settings; windows.value = result.policy.windows_alpha;
  configPath.value = result.configuration_path;
  domains.value = policy.value.allowed_domains.join("\n");
  paths.value = policy.value.tool_read_paths.join("\n"); loaded.value = true;
}); });
const lines = (value: string) => value.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
async function save() {
  if (policy.value.mode === "off" && !window.confirm("关闭沙箱后，批准的命令/测试/本地 MCP 将拥有当前用户权限，可能访问工作区外文件和网络。确认关闭？")) return;
  await perform(async () => {
    await api.saveSandbox({ ...policy.value, allowed_domains: lines(domains.value), tool_read_paths: lines(paths.value) }, policy.value.mode === "off");
    status.value = null;
    message.value = "已保存，后续工具调用生效。";
  });
}
async function probe() { await perform(async () => { status.value = null; status.value = await api.probeSandbox(); }); }
async function install() {
  if (!window.confirm("将安装 Windows alpha 沙箱：创建 srt-sandbox 专用账户、用户组和系统级 WFP 网络规则，需要 UAC 管理员确认。不会为普通命令自动提权，也不会安装 TLS 根证书。继续？")) return;
  await perform(async () => {
    status.value = null;
    const result = await api.installSandbox();
    if (result.cancelled) { message.value = "安装已取消。"; return; }
    status.value = await api.probeSandbox();
    message.value = "安装结束，已检查启动状态。";
  });
}
</script>

<template>
  <Modal title="执行沙箱" @close="$emit('close')">
    <div class="sandbox-settings">
      <div class="sandbox-intro">
        <p>限制命令、测试和本地 MCP 的文件与网络访问。</p>
        <span v-if="windows" class="sandbox-badge">Windows · 实验功能</span>
      </div>
      <label>执行模式
        <select v-model="policy.mode" :disabled="busy || !loaded">
          <option value="required">默认使用沙箱</option>
          <option value="off">不使用沙箱（风险较高）</option>
        </select>
      </label>
      <p class="sandbox-hint" v-if="policy.mode === 'required'">不可用时停止，不会自动绕过。</p>
      <p class="sandbox-warning" v-else>批准的程序将拥有当前用户权限，不受沙箱隔离。</p>
      <div class="sandbox-exception">
        <label class="sandbox-checkbox"><input v-model="policy.allow_approved_host_execution" type="checkbox" :disabled="busy || !loaded" />允许申请单次沙箱外执行</label>
        <p class="sandbox-hint">每次单独审批；批准后，该次操作不受沙箱保护。</p>
      </div>
      <details class="sandbox-advanced">
        <summary>高级设置</summary>
        <div class="sandbox-advanced-body">
          <label>允许联网的域名<textarea v-model="domains" rows="3" :disabled="busy || !loaded" placeholder="registry.npmjs.org&#10;pypi.org" /></label>
          <p class="sandbox-hint">每行一个，留空禁止通过代理联网。</p>
          <label>额外只读工具目录<textarea v-model="paths" rows="2" :disabled="busy || !loaded" placeholder="每行一个绝对路径" /></label>
          <p class="sandbox-hint">仅授予读取权限，不要添加整个用户目录。</p>
          <p class="sandbox-path">配置文件：{{ configPath || '读取中…' }}</p>
          <p class="sandbox-hint">保护命令进程，不是虚拟机或全盘读取白名单；原生文件工具仍使用路径授权，远程 MCP 不在本机沙箱内。Windows alpha 后端不能阻断系统 DNS 查询。</p>
        </div>
      </details>
      <div class="sandbox-actions"><button :disabled="busy || !loaded" @click="save">保存设置</button><button :disabled="busy || !loaded" @click="probe">检查环境</button><button v-if="windows" :disabled="busy || !loaded" @click="install">安装沙箱…</button></div>
      <p v-if="busy" class="sandbox-hint" role="status">正在处理…</p>
      <p v-if="message" class="sandbox-hint" role="status">{{ message }}</p>
      <div v-if="error" class="sandbox-result sandbox-result-failed" role="alert"><strong>操作未完成</strong><details class="sandbox-diagnostics"><summary>查看错误详情</summary><pre>{{ error }}</pre></details></div>
      <section v-if="status" class="sandbox-result" :class="{ 'sandbox-result-failed': !status.ready }" aria-label="沙箱检查结果" role="status">
        <strong>{{ !status.ready ? '尚未就绪' : status.startupVerified ? '启动自检通过' : '依赖检查通过' }}</strong>
        <p class="sandbox-hint">{{ !status.ready ? '未通过检查，默认命令不会执行。' : status.startupVerified ? '已验证启动与网络隔离，不代表你的任务已执行。' : '尚未验证实际进程启动。' }}</p>
        <details v-if="status.errors.length || status.warnings.length" class="sandbox-diagnostics"><summary>查看诊断详情（{{ status.errors.length + status.warnings.length }}）</summary><pre v-for="item in [...status.errors, ...status.warnings]" :key="item">{{ item }}</pre></details>
      </section>
    </div>
  </Modal>
</template>

<style scoped>
.sandbox-settings { display: grid; gap: 14px; min-width: 0; }
.sandbox-settings p { margin: 0; line-height: 1.5; }
.sandbox-intro { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; color: #b8bbc0; font-size: 14px; }
.sandbox-badge { padding: 3px 8px; border: 1px solid #625846; border-radius: 6px; color: #d8c29f; font-size: 12px; white-space: nowrap; }
.sandbox-settings label { display: grid; gap: 8px; }
.sandbox-exception { padding: 14px; border: 1px solid #404247; border-radius: 10px; display: grid; gap: 7px; }
.sandbox-settings .sandbox-checkbox { display: flex; align-items: center; gap: 10px; }
.sandbox-checkbox input { width: 16px; height: 16px; flex: 0 0 16px; margin: 0; }
.sandbox-settings select, .sandbox-settings textarea { width: 100%; box-sizing: border-box; border: 1px solid #505050; border-radius: 8px; padding: 12px; background: #181818; color: #ededed; font: inherit; }
.sandbox-settings textarea { resize: vertical; }
.sandbox-hint { color: #a4a7ae; font-size: 13px; }
.sandbox-warning { color: #ffc7a1; }
.sandbox-advanced { border-block: 1px solid #3a3c41; padding: 12px 0; }
.sandbox-settings summary { cursor: pointer; font-size: 14px; color: #c8cbd2; }
.sandbox-advanced-body { display: grid; gap: 10px; padding-top: 15px; }
.sandbox-path { overflow-wrap: anywhere; font-size: 12px; color: #8f939b; }
.sandbox-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 10px; }
.sandbox-actions button { white-space: nowrap; }
.sandbox-result { display: grid; gap: 7px; background: #202b25; border: 1px solid #3d5746; padding: 14px; border-radius: 10px; }
.sandbox-result strong { font-size: 15px; color: #b9d9c0; }
.sandbox-result-failed { background: #302722; border-color: #685243; }
.sandbox-result-failed strong { color: #ffc7a1; }
.sandbox-diagnostics pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 240px; overflow: auto; font-size: 12px; color: #b9bcc4; }
</style>
