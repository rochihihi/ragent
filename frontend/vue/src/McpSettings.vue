<script setup lang="ts">
import { onMounted, ref } from "vue";
import Modal from "./Modal.vue";
import { sandboxHeaders } from "../../src/api";
type Config = { transport?: string; command?: string; args?: string[]; cwd?: string; url?: string; headers_env?: Record<string, string>; enabled?: boolean };
type Server = { name: string; config: Config; has_env: boolean; configuration_fingerprint?: string };
type Tool = { name: string; description?: string; inputSchema?: unknown };
const props = defineProps<{ repoRoot: string }>();
const emit = defineEmits<{ close: [] }>();
const servers = ref<Server[]>([]), configPath = ref("");
const name = ref(""), editing = ref<string | null>(null), transport = ref("stdio"), command = ref(""), args = ref("[]"), cwd = ref(""), url = ref(""), headers = ref("{}"), enabled = ref(true), root = ref(props.repoRoot);
const busy = ref(false), notice = ref(""), tools = ref<Tool[]>([]), tested = ref("");
async function request(path = "", method = "GET", body?: unknown, host = false) {
  const response = await fetch(`/mcp-servers${path}`, { method, headers: host ? await sandboxHeaders() : { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await response.json(); if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "MCP 请求失败"); return data;
}
async function load() { const data = await request() as { servers: Server[]; path: string }; servers.value = data.servers; configPath.value = data.path; }
async function perform(action: () => Promise<void>) { busy.value = true; notice.value = ""; try { await action(); } catch (e) { notice.value = (e as Error).message; } finally { busy.value = false; } }
function edit(server?: Server) { editing.value = server?.name || null; name.value = server?.name || ""; const c = server?.config || {}; transport.value = c.transport || "stdio"; command.value = c.command || ""; args.value = JSON.stringify(c.args || [], null, 2); cwd.value = c.cwd || ""; url.value = c.url || ""; headers.value = JSON.stringify(c.headers_env || {}, null, 2); enabled.value = c.enabled !== false; notice.value = ""; }
async function save() {
  if (!name.value.trim() || name.value.includes("::") || name.value === "builtin") throw new Error("请填写有效服务名称，不含 ::，不能使用 builtin");
  const config: Config = { transport: transport.value, enabled: enabled.value };
  if (transport.value === "stdio") { const parsed = JSON.parse(args.value); if (!Array.isArray(parsed) || !parsed.every(value => typeof value === "string")) throw new Error("参数必须是字符串 JSON 数组"); config.command = command.value; config.args = parsed; if (cwd.value) config.cwd = cwd.value; }
  else { const parsed = JSON.parse(headers.value); if (!parsed || Array.isArray(parsed) || typeof parsed !== "object" || !Object.values(parsed).every(value => typeof value === "string")) throw new Error("认证映射必须是 JSON 对象"); config.url = url.value; config.headers_env = parsed as Record<string, string>; }
  await request(`/${encodeURIComponent(name.value)}`, "PUT", { config }); await load(); editing.value = name.value; notice.value = "已保存。配置变化后旧执行授权失效。";
}
async function test(server: Server, host = false) {
  tools.value = []; tested.value = ""; if (!root.value.trim()) throw new Error("请填写测试工作目录");
  const target = server.config.transport === "streamable_http" ? server.config.url : `${server.config.command}\n参数：${JSON.stringify(server.config.args || [])}`;
  const boundary = host ? "本次在沙箱外启动服务，以当前用户权限运行，可访问工作区外文件和网络。不会关闭其他调用的沙箱，也不会保存此授权。" : "按当前默认沙箱策略启动；失败不会自动切换为沙箱外执行。";
  if (!window.confirm(`测试 ${server.name} 会启动外部程序或访问网络，不是只读保证。仅测试可信服务。\n${boundary}\n完整命令/地址：${target}\n工作目录：${server.config.cwd || root.value}\n继续？`)) return;
  const result = await request(`/${encodeURIComponent(server.name)}/test`, "POST", { confirmed: true, repo_root: root.value, execution_mode: host ? "host" : "sandbox", confirm_host_execution: host, configuration_fingerprint: server.configuration_fingerprint }, host) as { tools: Tool[] };
  tools.value = result.tools; tested.value = server.name; notice.value = `连接成功${host ? '（本次沙箱外执行）' : ''}，发现 ${result.tools.length} 个工具。`;
}
function removeServer(server: Server) {
  if (!window.confirm(`删除 ${server.name} 的配置？不会卸载外部程序。`)) return;
  void perform(async () => { await request(`/${encodeURIComponent(server.name)}`, "DELETE"); await load(); if (editing.value === server.name) edit(); });
}
onMounted(() => void load().catch(e => { notice.value = e.message; }));
</script>
<template>
  <Modal title="MCP 服务管理" :busy="busy" :wide="true" @close="emit('close')">
    <p>此处管理外部 MCP 服务。项目文件使用原生文件工具，不需要内置 MCP 服务。外部服务可能执行程序、修改数据或联网，调用仍受审批控制。</p>
    <details><summary>配置保存说明</summary><p>服务设置由 RAgent 保存在当前 Windows 用户目录中，更换项目或更新程序后仍保留。</p><code>{{ configPath || '正在读取配置位置…' }}</code></details>
    <div class="form-actions"><button :disabled="busy" @click="edit()">添加服务</button><button :disabled="busy" @click="perform(load)">刷新</button></div>
    <article v-for="server in servers" :key="server.name" class="mcp-server-card">
      <strong>{{ server.name }}</strong> · {{ server.config.transport || 'stdio' }} · {{ server.config.enabled === false ? '已禁用' : '已启用' }}
      <p class="wrap-anywhere">{{ server.config.transport === 'streamable_http' ? server.config.url : server.config.command }}</p>
      <small v-if="server.has_env">已有私有环境变量，编辑时保留，不回显。</small>
      <div class="form-actions">
        <button :disabled="busy" @click="edit(server)">编辑</button>
        <button :disabled="busy" @click="perform(async () => { await request(`/${encodeURIComponent(server.name)}`, 'PUT', { config: { ...server.config, enabled: server.config.enabled === false } }); await load(); })">{{ server.config.enabled === false ? '启用' : '禁用' }}</button>
        <button :disabled="busy || server.config.enabled === false" @click="perform(() => test(server))">测试连接</button>
        <button v-if="server.config.transport !== 'streamable_http'" :disabled="busy || server.config.enabled === false" @click="perform(() => test(server, true))">本次沙箱外测试…</button>
        <button :disabled="busy" @click="removeServer(server)">删除</button>
      </div>
    </article>
    <p v-if="!servers.length">尚未添加外部服务。</p>
    <h3>{{ editing ? `编辑 ${editing}` : '新服务' }}</h3>
    <fieldset :disabled="busy" class="mcp-fields"><label>服务名称<input :disabled="editing !== null" v-model="name" /></label><label>连接方式<select v-model="transport"><option value="stdio">启动程序（stdio）</option><option value="streamable_http">连接地址（Streamable HTTP，本机或远程）</option></select></label><small>{{ transport === 'stdio' ? 'RAgent 启动配置的程序，通过标准输入输出通信。' : '连接已运行的 MCP 服务；支持本机 HTTP 地址，远程地址需 HTTPS。' }}</small><template v-if="transport === 'stdio'"><label>可执行程序<input v-model="command" placeholder="node.exe 或完整路径" /></label><label>完整参数（JSON 数组）<textarea v-model="args" rows="4" placeholder="['C:\\my-mcp\\server.js']" /></label><label>程序工作目录（可选）<input v-model="cwd" /></label></template><template v-else><label>MCP 地址<input v-model="url" placeholder="https://example.com/mcp" /></label><label>认证请求头 → 环境变量名（不输入密钥）<textarea v-model="headers" rows="3" placeholder="{&quot;Authorization&quot;:&quot;MY_MCP_AUTHORIZATION&quot;}" /></label><small>环境变量值如 Bearer …；需要在启动 RAgent 前设置。</small></template><label class="mcp-enabled"><input v-model="enabled" type="checkbox" /><span>启用服务</span></label><div class="form-actions"><button class="primary" :disabled="busy" @click="perform(save)">{{ busy ? '处理中…' : '保存服务' }}</button></div><label>连接测试工作目录<input v-model="root" placeholder="项目目录的绝对路径" /></label></fieldset>
    <p v-if="notice" class="notice">{{ notice }}</p><section v-if="tested"><h3>{{ tested }} 的工具</h3><details v-for="tool in tools" :key="tool.name"><summary>{{ tool.name }}</summary><p>{{ tool.description }}</p><pre class="vue-code">{{ JSON.stringify(tool.inputSchema, null, 2) }}</pre></details></section>
  </Modal>
</template>
