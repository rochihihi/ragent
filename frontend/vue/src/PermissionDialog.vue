<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { Session } from "../../src/api";
import { approvalActions } from "../../src/permissionActions";
const props = defineProps<{ permission: NonNullable<Session['pending_permission']>; busy: boolean }>();
const emit = defineEmits<{ decide: [approved: boolean, instruction?: string, scope?: string] }>();
const suggestion = ref("");
watch(() => props.permission.request_id, () => { suggestion.value = ""; });
const actions = computed(() => approvalActions(props.permission.decision));
const hostExecution = computed(() => actions.value.some(item => item.raw.execution_mode === "host"));
const title = computed(() => hostExecution.value ? "允许本次沙箱外执行？" : props.permission.destructive ? "允许永久删除？" : props.permission.command?.length ? "允许执行命令？" : "允许执行此操作？");
</script>
<template>
  <div class="permission-overlay" role="dialog" aria-modal="true" aria-labelledby="permission-title">
    <section class="permission-card" :class="{ 'permission-danger': permission.destructive }">
      <header><i>{{ permission.destructive ? '!' : '>_' }}</i><div><span>RAgent 请求权限</span><h2 id="permission-title">{{ title }}</h2></div></header>
      <div class="permission-content">
        <section class="permission-why"><small>为什么现在需要这一步</small><p>{{ permission.reason }}</p></section>
        <section v-if="hostExecution" class="permission-why"><small>注意：沙箱外执行</small><p>标记为 host 的操作及其子进程拥有当前用户权限，可访问工作区外文件和网络。仅批准这次，不关闭全局沙箱；不授予管理员权限。</p></section>
        <section class="permission-summary"><small>执行理由与已知边界</small><p>{{ permission.purpose || '执行以下具体操作' }}</p><em>{{ permission.impact || '请核对完整参数和目标；批准不等于操作安全。' }}</em><em>范围：{{ permission.scope || permission.path }}</em><em v-if="permission.recovery">恢复：{{ permission.recovery }}</em></section>
        <section v-if="permission.operation === 'mcp_call' && permission.command?.length" class="permission-command"><small>将启动的本地 MCP 服务完整命令</small><pre>{{ JSON.stringify(permission.command, null, 2) }}</pre><p>工作目录：{{ permission.path }}</p></section>
        <section v-if="actions.length" class="permission-batch"><small>即将执行的具体操作</small>
          <article v-for="item in actions" :key="item.number" class="permission-command">
            <h4>{{ item.number }}. {{ item.action }}</h4><p>{{ item.rationale }}</p>
            <p v-if="item.raw.execution_mode">执行边界：{{ item.raw.execution_mode === 'host' ? '沙箱外 · 当前用户权限（本次例外）' : '默认沙箱策略 · 不自动绕过' }}</p>
            <template v-if="item.command.length"><small>完整命令参数（JSON 数组，保留参数边界）</small><pre>{{ JSON.stringify(item.command, null, 2) }}</pre><p>工作目录：{{ permission.path }}</p></template>
            <details><summary>完整执行参数</summary><pre>{{ JSON.stringify(item.raw, null, 2) }}</pre></details>
          </article>
        </section>
        <section v-else class="permission-command"><small>{{ permission.command?.length ? '将执行的完整命令' : '将访问的完整路径' }}</small><strong>{{ permission.command?.length ? JSON.stringify(permission.command, null, 2) : permission.path }}</strong></section>
        <details class="permission-suggestion"><summary>需要调整执行方案？</summary><div><input v-model="suggestion" aria-label="调整要求" @keydown.enter.prevent="suggestion.trim() && emit('decide', false, suggestion.trim())" /><button :disabled="busy || !suggestion.trim()" @click="emit('decide', false, suggestion.trim())">调整方案</button></div></details>
      </div>
      <footer><button v-if="!hostExecution && (actions.length || permission.command?.length)" :disabled="busy" @click="emit('decide', true, '', 'session')">本会话允许此{{ actions.length > 1 ? '批操作' : '操作' }}</button><button :disabled="busy" @click="emit('decide', false)">拒绝</button><button class="primary" :disabled="busy" @click="emit('decide', true)">{{ busy ? '处理中…' : hostExecution ? '本次沙箱外执行' : '本次允许' }}</button></footer>
    </section>
  </div>
</template>
