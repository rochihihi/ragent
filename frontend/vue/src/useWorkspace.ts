import { ref, computed, watch, nextTick, onMounted, onUnmounted } from "vue";
import { api, type Session, type Event, type Provider } from "../../src/api";

export const providers: Array<[Provider, string]> = [["deepseek", "DeepSeek"], ["openai_official", "OpenAI 官方"], ["openai", "中转站"]];
export const efforts: Record<Provider, Array<[string, string]>> = {
  deepseek: [["low", "低"], ["high", "高"], ["max", "最高"]],
  openai: [["low", "低"], ["medium", "中"], ["high", "高"], ["xhigh", "超高"], ["max", "最高"]],
  openai_official: [["low", "低"], ["medium", "中"], ["high", "高"], ["xhigh", "超高"], ["max", "最高"]],
};
export const phases: Record<string, string> = { idle: "待命", running: "运行中", preparing_context: "整理上下文", waiting_model: "等待模型", retrying_model: "精简重试", executing_tool: "执行工具", waiting_permission: "等待允许", paused: "已暂停", completed: "已完成", failed: "已停止" };

export function useWorkspace() {
  const sessions = ref<Session[]>([]), activeId = ref<string | null>(null), active = ref<Session | null>(null);
  const events = ref<Event[]>([]), files = ref<string[]>([]), directories = ref<string[]>([]);
  const draft = ref(""), error = ref(""), sending = ref(false), busy = ref(false), pausePending = ref(false);
  const messages = ref<HTMLElement | null>(null), followLatest = ref(true);
  let revision = 0, disposed = false, timer: ReturnType<typeof setTimeout> | undefined;
  const changed = computed(() => new Set(active.value?.changed_files ?? []));
  const visibleMessages = computed(() => (active.value?.messages ?? []).filter((m, i, all) =>
    !m.content.startsWith("调整执行方案：") && !m.content.startsWith("关于当前待批准操作，请调整方案：") &&
    (!i || m.content !== all[i - 1].content || m.role !== all[i - 1].role)));
  const locked = computed(() => !active.value || ["running", "waiting_permission"].includes(active.value.status) || busy.value);
  async function perform<T>(fn: () => Promise<T>): Promise<T | undefined> {
    try { return await fn(); } catch (e) { error.value = (e as Error).message; }
  }
  async function loadSessions() {
    const result = await api.sessions();
    if (disposed) return;
    sessions.value = result;
    if (!activeId.value && result[0]) activeId.value = result[0].session_id;
  }
  async function refresh() {
    const id = activeId.value;
    if (!id || sending.value) return;
    const current = ++revision;
    const [s, e, tree] = await Promise.all([api.session(id), api.events(id), api.files(id)]);
    if (disposed || current !== revision || id !== activeId.value || sending.value) return;
    active.value = s; events.value = e; files.value = tree.files; directories.value = tree.directories ?? [];
    sessions.value = sessions.value.map(item => item.session_id === id ? s : item);
    if (s.status !== "running") pausePending.value = false;
  }
  watch(activeId, () => {
    ++revision; active.value = null; events.value = []; files.value = []; directories.value = [];
    pausePending.value = false; draft.value = ""; followLatest.value = true;
    void perform(refresh);
  });
  watch(() => active.value?.messages.length, async () => {
    if (followLatest.value) { await nextTick(); messages.value?.scrollTo({ top: messages.value.scrollHeight, behavior: "smooth" }); }
  });
  function trackScroll() {
    const node = messages.value;
    if (node) followLatest.value = node.scrollHeight - node.scrollTop - node.clientHeight < 120;
  }
  function latest() { followLatest.value = true; messages.value?.scrollTo({ top: messages.value.scrollHeight, behavior: "smooth" }); }
  async function send() {
    const id = activeId.value, content = draft.value.trim(), previous = active.value;
    if (!id || !content || sending.value || !previous || previous.status === "waiting_permission") return;
    sending.value = true; ++revision; draft.value = ""; followLatest.value = true;
    active.value = { ...previous, status: "running", activity: "preparing_context", messages: [...previous.messages, { role: "user", content }] };
    try { await api.send(id, content); }
    catch (e) { if (activeId.value === id) { active.value = previous; draft.value = content; error.value = (e as Error).message; } }
    finally { sending.value = false; await perform(refresh); }
  }
  async function control() {
    const s = active.value;
    if (!s || busy.value || pausePending.value) return;
    busy.value = true;
    await perform(async () => {
      if (s.status === "running") {
        const result = await api.pause(s.session_id);
        if (activeId.value === s.session_id) pausePending.value = result.status === "pausing";
      } else await api.resume(s.session_id);
      await refresh();
    });
    busy.value = false;
  }
  async function permission(approved: boolean, instruction = "", scope = "once") {
    const s = active.value, p = s?.pending_permission;
    if (!s || !p || busy.value) return;
    busy.value = true;
    await perform(async () => { await api.decidePermission(s.session_id, p.request_id, approved, instruction, scope); await refresh(); });
    busy.value = false;
  }
  async function settings(body: object) {
    const s = active.value;
    if (!s || locked.value) return;
    busy.value = true;
    await perform(async () => {
      const updated = await api.updateSettings(s.session_id, { ...s, ...body });
      if (activeId.value === s.session_id) active.value = updated;
      await loadSessions();
    });
    busy.value = false;
  }
  async function remove(id: string) {
    if (!confirm("删除这个对话及执行记录？不会删除项目文件。")) return;
    await perform(async () => {
      await api.deleteSession(id);
      if (id === activeId.value) activeId.value = null;
      await loadSessions();
    });
  }
  async function poll() {
    if (disposed) return;
    await perform(async () => { await loadSessions(); await refresh(); });
    if (!disposed) timer = setTimeout(poll, 1200);
  }
  onMounted(() => void poll());
  onUnmounted(() => { disposed = true; ++revision; clearTimeout(timer); });
  return { sessions, activeId, active, events, files, directories, changed, draft, error, sending, busy, pausePending,
    messages, followLatest, visibleMessages, locked, perform, loadSessions, refresh, trackScroll, latest, send, control, permission, settings, remove };
}
