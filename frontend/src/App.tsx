import { CSSProperties, FormEvent, PointerEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, Event, GitPanelData, Provider, ProviderState, Quota, Session, UpgradeCheck } from "./api";
import "./git.css";
import { Skills } from "./Skills";
import Markdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import { CodeBlock } from "./CodeBlock";
import remarkGfm from "remark-gfm";

const effort: Record<Provider, Array<[string, string]>> = {
  deepseek: [["low", "低"], ["high", "高"], ["max", "最高"]],
  openai: [["low", "低"], ["medium", "中"], ["high", "高"], ["xhigh", "超高"], ["max", "最高"]],
  openai_official: [["low", "低"], ["medium", "中"], ["high", "高"], ["xhigh", "超高"], ["max", "最高"]],
};
const actionNames: Record<string, string> = { created: "会话已创建", user_message: "收到任务", assistant_message: "完整回答", workspace_entry_created: "项目条目已创建", task_contract: "建立任务契约", intent_clarification: "澄清任务意图", intent_classifier_fallback: "意图分类安全降级", context_compaction_fallback: "上下文摘要恢复", context_compressed: "上下文已压缩", context_trimmed: "上下文已裁剪", steer_queued: "运行中纠正已排队", steer_applied: "已切换任务目标", plan: "制定计划", plan_resumed: "恢复执行计划", context_prepared: "上下文已准备", model_waiting: "等待模型", model_retrying: "精简重试", model_response: "模型响应", model_diagnostic: "调用诊断", duplicate_corrected: "自动纠正重复动作", decision: "Agent 决策", observation: "工具结果", patch: "补丁已应用", permission_requested: "请求权限", permission_revised: "调整执行方案", permission_approved: "权限已允许", permission_execution_failed: "已允许，执行失败", permission_recovered: "权限状态已恢复", permission_denied: "权限被拒绝", interrupted: "执行已中断", paused: "任务已暂停", failed: "任务停止", completed: "任务完成", settings_updated: "配置已更新" };

const activityText: Record<string, string> = {
  preparing_context: "整理上下文",
  waiting_model: "等待模型",
  retrying_model: "精简重试",
  executing_tool: "执行工具",
  waiting_permission: "等待允许",
  paused: "已暂停",
  completed: "已完成",
  failed: "已停止",
  idle: "待命",
};

type Modal = "new" | "api" | "acceptance" | "settings" | "folders" | "git" | "skills" | null;
type FileView = { path: string; changed: boolean; before: string | null; after: string; diff: string } | null;
type FileNode = { name: string; path: string; kind: "file" | "folder"; children: FileNode[] };

export function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [active, setActive] = useState<Session | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const [files, setFiles] = useState<string[]>([]);
  const [directories, setDirectories] = useState<string[]>([]);
  const [selectedDirectory, setSelectedDirectory] = useState("");
  const [newEntryKind, setNewEntryKind] = useState<"file" | "folder" | null>(null);
  const [newEntryPath, setNewEntryPath] = useState("");
  const [modal, setModal] = useState<Modal>(null);
  const [draft, setDraft] = useState("");
  const [fileView, setFileView] = useState<FileView>(null);
  const [error, setError] = useState("");
  const [permissionBusy, setPermissionBusy] = useState(false);
  const [permissionSuggestion, setPermissionSuggestion] = useState("");
  const [leftPane, setLeftPane] = useState(270);
  const [rightPane, setRightPane] = useState(330);
  const [showTrace, setShowTrace] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [showLatest, setShowLatest] = useState(false);
  const [clock, setClock] = useState(Date.now());
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const messagesRef = useRef<HTMLElement | null>(null);
  const followLatest = useRef(true);
  const refreshVersion = useRef(0);
  const selectedSession = useRef(activeId);
  selectedSession.current = activeId;
  const submitting = useRef(false);
  const [eventFloor, setEventFloor] = useState(0);

  const loadSessions = useCallback(async () => {
    const value = await api.sessions(); setSessions(value);
    if (!activeId && value[0]) setActiveId(value[0].session_id);
  }, [activeId]);
  const refresh = useCallback(async () => {
    if (!activeId || submitting.current) return;
    const version = ++refreshVersion.current;
    const [session, nextEvents, tree] = await Promise.all([api.session(activeId), api.events(activeId), api.files(activeId)]);
    if (version !== refreshVersion.current || selectedSession.current !== activeId || submitting.current) return;
    setActive(session); setEvents(nextEvents); setFiles(tree.files); setDirectories(tree.directories || []);
    setSessions((current) => current.map((item) => item.session_id === session.session_id ? session : item));
  }, [activeId]);

  useEffect(() => { loadSessions().catch((e) => setError(e.message)); }, [loadSessions]);
  useEffect(() => { refresh().catch((e) => setError(e.message)); }, [refresh]);
  useEffect(() => {
    if (!active || active.status !== "running") return;
    const timer = window.setInterval(() => refresh().catch((e) => setError(e.message)), 900);
    return () => window.clearInterval(timer);
  }, [active, refresh]);
  useEffect(() => {
    if (!active || active.status !== "running") return;
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active?.status]);
  useEffect(() => {
    followLatest.current = true;
    setShowLatest(false);
    setRunStartedAt(null);
    setEventFloor(0);
    setSelectedDirectory("");
    setNewEntryKind(null);
    setNewEntryPath("");
  }, [activeId]);
  useEffect(() => {
    if (active?.status !== "running") setRunStartedAt(null);
  }, [active?.status]);
  useEffect(() => { setPermissionSuggestion(""); }, [active?.pending_permission?.request_id]);
  useEffect(() => {
    if (!followLatest.current) return;
    const frame = window.requestAnimationFrame(() => {
      const node = messagesRef.current;
      if (node) node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [activeId, active?.messages.length]);

  function trackMessageScroll() {
    const node = messagesRef.current;
    if (!node) return;
    const atLatest = node.scrollHeight - node.scrollTop - node.clientHeight < 120;
    followLatest.current = atLatest;
    setShowLatest(!atLatest);
  }
  function scrollToLatest() {
    followLatest.current = true;
    setShowLatest(false);
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: "smooth" });
  }

  async function send(event: FormEvent) {
    event.preventDefault(); if (!activeId || !draft.trim()) return;
    if (submitting.current) return;
    submitting.current = true;
    ++refreshVersion.current;
    const content = draft.trim();
    setEventFloor(events.at(-1)?.sequence ?? 0);
    setDraft(""); setRunStartedAt(Date.now());
    setActive((s) => s ? { ...s, status: "running", activity: "preparing_context", plan: [], messages: [...s.messages, { role: "user", content }] } : s);
    try { await api.send(activeId, content); }
    catch (e) { if (selectedSession.current === activeId) { setDraft(content); setError((e as Error).message); } }
    finally { submitting.current = false; await refresh(); }
  }
  async function remove(id: string) {
    if (!confirm("删除这个对话及其轨迹？")) return;
    try { await api.deleteSession(id); if (activeId === id) { setActiveId(null); setActive(null); } await loadSessions(); } catch (e) { setError((e as Error).message); }
  }
  async function openFile(path: string) { if (activeId) try { setFileView(await api.fileChange(activeId, path)); } catch (e) { setError((e as Error).message); } }
  async function createEntry(event: FormEvent) {
    event.preventDefault();
    if (!activeId || !newEntryKind || !newEntryPath.trim()) return;
    try {
      const name = newEntryPath.trim().replace(/^[/\\]+/, "");
      const path = selectedDirectory ? `${selectedDirectory}/${name}` : name;
      await api.createProjectEntry(activeId, newEntryKind, path);
      setNewEntryKind(null); setNewEntryPath(""); await refresh();
    } catch (e) { setError((e as Error).message); }
  }
  async function decidePermission(approved: boolean, instruction = "", scope = "once") {
    if (!active?.pending_permission || permissionBusy) return;
    const previous = active;
    const requestId = active.pending_permission.request_id;
    setPermissionBusy(true);
    if (approved) setRunStartedAt(Date.now());
    setActive((session) => session ? { ...session, pending_permission: null, status: approved ? "running" : "idle" } : session);
    try {
      await api.decidePermission(previous.session_id, requestId, approved, instruction, scope);
      await refresh();
    } catch (e) {
      setActive(previous);
      setError((e as Error).message);
    } finally {
      setPermissionBusy(false);
    }
  }
  function resizePane(side: "left" | "right", event: PointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    const bounds = event.currentTarget.parentElement!.getBoundingClientRect();
    const move = (next: globalThis.PointerEvent) => side === "left"
      ? setLeftPane(Math.min(440, Math.max(210, next.clientX - bounds.left)))
      : setRightPane(Math.min(520, Math.max(270, bounds.right - next.clientX)));
    const stop = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", stop); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", stop, { once: true });
  }

  const changed = useMemo(() => new Set(active?.changed_files ?? []), [active]);
  const visibleMessages = useMemo(() => (active?.messages ?? []).filter(
    (message, index, all) => !message.content.startsWith("调整执行方案：") && !message.content.startsWith("关于当前待批准操作，请调整方案：") && (index === 0 || message.content !== all[index - 1].content || message.role !== all[index - 1].role),
  ), [active?.messages]);
  const activityEvents = useMemo(
    () => events.filter((item) => !["settings_updated", "memory_updated", "plan_updated"].includes(item.event_type)),
    [events],
  );
  const contextUsage = useMemo(
    () => contextMetrics(
      events,
      active?.context_estimated_tokens || 0,
      active?.context_actual_input_tokens,
      active?.context_limit_tokens,
      active?.status === "running",
    ),
    [events, active?.context_estimated_tokens, active?.context_actual_input_tokens, active?.context_limit_tokens, active?.status],
  );
  const fileTree = useMemo(() => buildFileTree(files, directories), [files, directories]);
  return <div className="app-shell">
    <Titlebar />
    <div className={`workspace${showTrace ? " with-trace" : ""}${showSidebar ? " with-sidebar" : ""}`} style={{ "--left-pane": `${leftPane}px`, "--right-pane": `${rightPane}px` } as CSSProperties}>
      <aside className="sidebar">
        <div className="side-actions"><button className="primary" onClick={() => setModal("new")}>＋ 新建对话</button><button className="icon" title="Git 工作区" disabled={!active} onClick={() => setModal("git")}>Git</button><button className="icon" title="升级验收" onClick={() => setModal("acceptance")}>验收</button><button className="icon" title="API 配置" onClick={() => setModal("api")}>设置</button></div>
        <SectionLabel>工作区会话</SectionLabel>
        <div className="session-list">{sessions.map((item) => <div className={`session-row ${item.session_id === activeId ? "active" : ""}`} key={item.session_id}><button onClick={() => setActiveId(item.session_id)}><strong>{item.title || "未命名任务"}</strong><small>{item.status} · {item.step} 步</small></button><button className="delete" onClick={() => remove(item.session_id)}>×</button></div>)}</div>
        <section className="project-panel">
          <header><div><span>项目文件</span><strong>{active ? projectName(active.repo_root) : "尚未选择仓库"}</strong></div><nav><button disabled={!active || active.status === "running"} title={`在${selectedDirectory || "项目根目录"}中新建文件`} onClick={() => { setNewEntryKind("file"); setNewEntryPath(""); }}>新建文件</button><button disabled={!active || active.status === "running"} title={`在${selectedDirectory || "项目根目录"}中新建文件夹`} onClick={() => { setNewEntryKind("folder"); setNewEntryPath(""); }}>新建文件夹</button><b>{files.length}</b></nav></header>
          {newEntryKind && <form className="new-project-entry" onSubmit={createEntry}><div className="new-entry-location"><span>创建位置</span><strong>{selectedDirectory || "项目根目录"}</strong>{selectedDirectory && <button type="button" onClick={() => setSelectedDirectory("")}>改到根目录</button>}</div><input autoFocus value={newEntryPath} onChange={(event) => setNewEntryPath(event.target.value)} placeholder={newEntryKind === "file" ? "输入文件名，例如 app.py" : "输入文件夹名称"} /><button className="primary" disabled={!newEntryPath.trim()}>创建</button><button type="button" onClick={() => setNewEntryKind(null)}>×</button></form>}
          <div className="file-tree">{fileTree.map((node) => <FileTreeNode changed={changed} key={node.path} node={node} onOpen={openFile} onSelectFolder={setSelectedDirectory} selectedFolder={selectedDirectory} />)}</div>
        </section>
      </aside>
      <div className="pane-resizer left" onPointerDown={(event) => resizePane("left", event)} />
      <main className="conversation">
        <header className="hero" key={`hero-${activeId || "empty"}`}><button aria-label="切换侧栏" aria-expanded={showSidebar} onClick={() => setShowSidebar(!showSidebar)}>☰</button><div className="conversation-heading"><h1>{active?.title || "开始一个编码任务"}</h1><p>{active ? `${active.provider} · ${active.model} · ${active.verification_mode}` : "选择项目，然后用自然语言要求 Agent 阅读、修改并验证代码。"}</p></div><div className="conversation-actions"><button disabled={!active || active.status === "running" || active.status === "waiting_permission"} onClick={() => setModal("skills")}>技能</button><button disabled={!active} onClick={() => setModal("settings")}>会话设置</button></div><button aria-expanded={showTrace} onClick={() => setShowTrace(!showTrace)}>执行记录</button></header>
        <section ref={messagesRef} onScroll={trackMessageScroll} className="messages" key={`messages-${activeId || "empty"}`}>
          {visibleMessages.length ? visibleMessages.map((message, index) => <article className={message.role} style={{ "--message-index": Math.min(index, 8) } as CSSProperties} key={`${message.role}-${index}`}>{message.role === "assistant" && <img className="assistant-avatar" src="/studio-brand" alt="" />}<MessageContent content={message.content} /></article>) : <div className="welcome"><img src="/studio-brand" alt="" /><h2>从项目开始</h2><p>让 RAgent 阅读代码、修改文件，或检查测试结果。</p></div>}
          {active?.status === "running" && <WorkingStatus session={active} events={events.filter((event) => event.sequence > eventFloor)} now={clock} runStartedAt={runStartedAt} />}
          {active && ["paused", "failed"].includes(active.status) && <RuntimeDiagnostic session={active} events={events} />}
        </section>
        {showLatest && <button className="scroll-latest" onClick={scrollToLatest}><span>↓</span>回到最新</button>}
        <form className="composer" onSubmit={send}><div className="compose-card"><textarea value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={active?.status === "running" ? "输入纠正或追加要求，将在安全边界切换…" : "给 RAgent 一个任务…"} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }} /><button type="submit" disabled={!active || !draft.trim()}>{active?.status === "running" ? "追加" : "发送"}</button><ComposerMeta key={modal === "api" ? "api" : "ready"} session={active} onChanged={refresh} /></div></form>
      </main>
      <div className="pane-resizer right" onPointerDown={(event) => resizePane("right", event)} />
      <aside className="trace"><header><div className="trace-heading"><strong>执行记录</strong><small>{activityEvents.length} 条记录</small></div><span className={`state ${active?.status || "idle"}`}><i />{activityText[active?.activity || active?.status || "idle"] || "待命"}</span></header><div className="activity-feed">{activityEvents.slice().reverse().map((item) => <details className={`activity-entry event-${item.event_type}`} key={item.sequence} open={["failed", "paused"].includes(item.event_type)}><summary><span className="activity-head"><strong>{actionNames[item.event_type] || item.event_type}</strong><time>{new Date(item.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time></span><small>{summary(item)}</small></summary><pre>{JSON.stringify(item.payload, null, 2)}</pre></details>)}</div><footer className="trace-summary"><div><span>执行步骤</span><strong>{active?.step || 0}</strong></div><div><span>修改文件</span><strong>{active?.changed_files.length || 0}</strong></div><div className="context-usage"><span>最近一次模型上下文</span><strong>{contextUsage.remaining}</strong><small>{contextUsage.detail}<br />{contextUsage.actualDetail}</small><i><b style={{ width: `${Math.min(contextUsage.percent || 0, 100)}%` }} /></i></div><div className="context-compression"><span>上下文处理</span><strong>{contextUsage.compressions ? `${contextUsage.compressions} 次` : "未触发"}</strong>{contextUsage.latestCompression && <small>{contextUsage.latestCompression}</small>}</div><div><span>当前阶段</span><strong className={active?.status || "idle"}>{activityText[active?.activity || active?.status || "idle"] || "待命"}</strong></div></footer></aside>
    </div>
    {error && <div className="toast" onClick={() => setError("")}>{error}</div>}
    {modal === "new" && <NewSession onClose={() => setModal(null)} onCreated={async (id) => { setModal(null); await loadSessions(); setActiveId(id); }} />}
    {modal === "skills" && active && <Skills id={active.session_id} onClose={() => setModal(null)} />}
    {modal === "api" && <ApiSettings onClose={() => setModal(null)} />}
    {modal === "acceptance" && <UpgradeAcceptance onClose={() => setModal(null)} onBack={() => setModal("api")} />}
    {modal === "settings" && active && <SessionSettings session={active} onClose={() => setModal(null)} onSaved={(updated) => { setActive(updated); setSessions((current) => current.map((item) => item.session_id === updated.session_id ? updated : item)); setModal(null); }} />}
    {modal === "git" && active && <GitPanel session={active} onClose={() => setModal(null)} />}
    {fileView && <FileDrawer value={fileView} onClose={() => setFileView(null)} />}
    {active?.pending_permission && <PermissionDialog permission={active.pending_permission} busy={permissionBusy} suggestion={permissionSuggestion} onSuggestion={setPermissionSuggestion} onDecision={decidePermission} />}
  </div>;
}

function Titlebar() {
  return <header className="titlebar">
    <div className="titlebar-drag pywebview-drag-region" onDoubleClick={() => window.pywebview?.api?.maximize?.()}><img src="/studio-brand" alt="RAgent" /></div>
    <div className="window-actions">
      <button aria-label="最小化" title="最小化" onClick={() => window.pywebview?.api?.minimize?.()}><svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M1 6h10" fill="none" stroke="currentColor" /></svg></button>
      <button aria-label="最大化或还原" title="最大化或还原" onClick={() => window.pywebview?.api?.maximize?.()}><svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="1.5" y="1.5" width="9" height="9" fill="none" stroke="currentColor" /></svg></button>
      <button aria-label="关闭" title="关闭" onClick={() => window.pywebview?.api?.close?.()}><svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="m1.5 1.5 9 9m0-9-9 9" fill="none" stroke="currentColor" /></svg></button>
    </div>
  </header>;
}
function PermissionDialog({ permission, busy, suggestion, onSuggestion, onDecision }: { permission: NonNullable<Session["pending_permission"]>; busy: boolean; suggestion: string; onSuggestion: (value: string) => void; onDecision: (approved: boolean, instruction?: string, scope?: string) => void }) { const pathInfo = explainPathPermission(permission); const fallback = explainPermission(permission.command); const purpose = permission.purpose || pathInfo?.purpose || fallback.purpose; const impact = permission.impact || pathInfo?.impact || fallback.impact; const risk = permission.risk || pathInfo?.risk || fallback.risk; const destructive = permission.destructive || pathInfo?.destructive || false; const rawRecommendation = permission.recommendation; const recommendation = rawRecommendation?.startsWith("建议拒绝") ? "系统暂时无法自动判断全部影响。请核对完整命令和目标路径；确认与当前任务一致时可本次允许，看不懂或范围不明确时再拒绝并要求调整。" : rawRecommendation; const scope = permission.scope; const why = permission.reason; return <div className="permission-overlay" role="dialog" aria-modal="true" aria-labelledby="permission-title"><section className={`permission-card${destructive ? " permission-danger" : ""}`}><header><i>{destructive ? "!" : permission.access === "execute" ? ">_" : "⌘"}</i><div><span>RAgent 请求权限</span><h2 id="permission-title">{destructive ? "允许永久删除？" : permission.access === "execute" ? "允许执行命令？" : permission.access === "write" ? "允许修改文件？" : permission.access === "action" ? "允许执行此操作？" : "允许读取文件？"}</h2></div></header><div className="permission-content">{why && <section className="permission-why"><small>为什么现在需要这一步</small><p>{why}</p></section>}<section className={`permission-summary risk-${risk}`}><small>批准后具体会发生什么</small><p>{purpose}</p><em>{impact}</em>{scope && <em><b>范围：</b>{scope}</em>}{recommendation && <b className="permission-advice">判断提示：{recommendation}</b>}</section><div className="permission-command"><small>{permission.command?.length ? "将执行的完整命令" : "将访问的完整路径"}</small><strong>{permission.command?.length ? permission.command.join(" ") : permission.path}</strong></div><details className="permission-suggestion"><summary>需要调整执行方案？</summary><div><input aria-label="告诉 RAgent 如何调整" id="permission-suggestion" value={suggestion} onChange={(event) => onSuggestion(event.target.value)} placeholder="" onKeyDown={(event) => { if (event.key === "Enter" && suggestion.trim()) { event.preventDefault(); onDecision(false, suggestion.trim()); } }} /><button disabled={busy || !suggestion.trim()} onClick={() => onDecision(false, suggestion.trim())}>调整方案</button></div></details></div><footer>{(permission.decision || permission.command?.length > 0) && <button disabled={busy} onClick={() => onDecision(true, "", "session")}>本会话允许此类操作</button>}<button disabled={busy} onClick={() => onDecision(false)}>{busy ? "处理中…" : "拒绝"}</button><button disabled={busy} className={destructive ? "primary destructive" : "primary"} onClick={() => onDecision(true)}>{busy ? "正在处理…" : !permission.decision && !permission.command?.length ? "允许此路径（本会话）" : destructive ? "本次允许删除" : "本次允许"}</button></footer></section></div>; }
function explainPathPermission(permission: NonNullable<Session["pending_permission"]>) { if (permission.command?.length || !permission.path) return null; const destructive = /(?:永久删除|删除.*全部|清空|移除.*文件)/.test(permission.reason); if (destructive) return { destructive: true, purpose: `授权 RAgent 删除“${permission.path}”内本次任务指定的全部文件和子文件夹。点击允许后，Agent 会继续执行删除，而不只是查看该目录。`, impact: `该范围内的源码、配置、隐藏文件和未提交内容都可能被删除；不会因此获得 ${permission.path} 之外的权限。`, scope: `仅限“${permission.path}”及其子项。请确认这里确实是你想清空的目录。`, recovery: "这是高风险操作。未由 Git 跟踪、未提交或没有备份的文件通常无法由 RAgent 恢复，也不保证进入回收站。", reason: permission.reason, risk: "high" }; if (permission.access === "read") return { destructive: false, purpose: `读取“${permission.path}”中的文件和目录信息。`, impact: "只读取内容，不运行程序、不修改文件。", scope: `仅限“${permission.path}”及其子项。`, recovery: "只读操作不会产生需要恢复的更改。", reason: permission.reason, risk: "low" }; if (permission.access === "write") return { destructive: false, purpose: `允许 RAgent 在“${permission.path}”中创建或修改当前任务需要的文件。`, impact: "可能改变该目录中的文件内容，但不会自动获得其他位置的权限。", scope: `仅限“${permission.path}”及其子项。`, recovery: "Git 已跟踪的改动通常可以撤销；未跟踪文件是否可恢复取决于是否有备份。", reason: permission.reason, risk: permission.risk || "medium" }; return null; }
function explainPermission(command: string[]) {
  const original = command.join(" ");
  const text = original.toLowerCase();
  const target = command.at(-1) || "目标文件";
  const powershellReadOnly = /^(?:powershell|pwsh)(?:\.exe)?\b/i.test(original) && !/\b(?:Set-Content|Add-Content|Remove-Item|Move-Item|Copy-Item|New-Item|Start-Process)\b/i.test(original);
  if (powershellReadOnly && /\bSelect-String\b/i.test(original)) { let file = original.match(/-Path\s+['"]?([^'";]+)/i)?.[1]?.trim() || "指定文件"; if (file.startsWith("$")) { const variable = file.slice(1).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); file = original.match(new RegExp(`\\$${variable}\\s*=\\s*['\"]([^'\"]+)`))?.[1] || file; } return { purpose: `在“${file}”中搜索代码结构和关键词，并显示匹配内容与行号，帮助定位需要修改的位置。`, impact: "这是只读代码搜索：不会运行该程序、不会修改或删除文件，也不会更改系统设置。", reason: "Agent 需要先找到界面、函数和控件对应的源码位置，才能准确修改而不破坏无关代码。", risk: "low" }; }
  if (powershellReadOnly && /\b(?:Get-Content|Get-ChildItem|Test-Path)\b/i.test(original)) return { purpose: "读取文件、列出目录或检查路径是否存在，以了解项目结构和当前代码。", impact: "这是只读检查：不会运行项目、修改文件、安装软件或更改系统设置。", reason: "Agent 需要先确认项目当前状态，再决定下一步修改。", risk: "low" };
  if (/go(?:\.exe)?\s+run\s+/i.test(original)) return { purpose: `使用电脑上已经安装的 Go 临时编译并运行“${target}”。如果它是图形程序，运行后会弹出该程序窗口。`, impact: "会启动一个独立的 Go 程序进程，并写入 Go 的临时编译缓存；不会安装软件，也不会修改当前项目源文件。关闭新程序窗口即可结束它。", reason: `你要求打开这个 Go 程序；必须实际运行“${target}”，只读取代码无法显示它的界面。`, risk: "medium" };
  if (/python(?:\.exe)?\s+[^\s]+\.py/i.test(original)) return { purpose: `使用电脑上已有的 Python 运行“${target}”。如果它包含图形界面，运行后会弹出新窗口。`, impact: "会启动一个独立的 Python 程序进程；程序运行期间可能按其代码访问项目文件，但这条命令本身不会安装软件。", reason: `你要求打开并查看该程序，因此需要实际运行“${target}”。`, risk: "medium" };
  if (text.startsWith("cmd /c if exist ") && text.includes("where go") && text.includes("go version") && text.includes("dir /a")) { const location = original.match(/if exist\s+([^\s(]+)/i)?.[1] || "指定位置"; return { purpose: `检查 ${location} 是否存在，并确认系统能否找到 Go 及其版本。`, impact: "只会读取目录、PATH 和版本信息，不会安装 Go、修改文件或更改系统设置。", reason: "需要确认 Go 是否已经安装以及安装位置，避免重复安装或运行错误版本。", risk: "low" }; }
  if (/cmd\s+\/c\s+where\s+\w+\s+&&\s+\w+\s+version/.test(text)) return { purpose: "检查电脑是否能找到目标工具，并读取已安装版本。", impact: "只读取系统环境信息，不会安装软件或修改文件。", reason: "需要先确认任务依赖的工具能够正常使用。", risk: "low" };
  if (text.includes("winget") && text.includes("install")) return { purpose: "使用 Windows 包管理器下载并安装任务所需的软件。", impact: "会联网下载并安装软件，还可能修改 PATH 等系统环境配置。", reason: "当前电脑缺少运行目标程序必需的软件。", risk: "high" };
  return { purpose: "执行 Agent 为当前任务提议的操作。", impact: "无法自动确认完整影响，请查看下面的原始命令后再决定。", reason: "该命令用于继续当前任务。", risk: "unknown" };
}
function RuntimeDiagnostic({ session, events }: { session: Session; events: Event[] }) {
  const boundary = events.reduce((last, event, index) => event.event_type === "user_message" || event.event_type === "steer_applied" ? index : last, -1);
  const currentEvents = boundary >= 0 ? events.slice(boundary) : [];
  const latest = (type: string) => [...currentEvents].reverse().find((event) => event.event_type === type);
  const diagnostic = latest("model_diagnostic");
  const retry = latest("model_retrying");
  const context = latest("context_prepared");
  const failure = `${session.pause_reason || ""} ${session.failure_reason || ""}`;
  const modelRelated = diagnostic || retry || /model|模型|timeout|protocol|http|connect/i.test(failure);
  if (!modelRelated) return null;
  const payload = diagnostic?.payload || {};
  const source = String(payload.source || (retry ? "中转站、网络代理或上游连接" : "暂时无法唯一确定"));
  const confidence = String(payload.confidence || (retry ? "中" : "低"));
  const evidence = String(payload.evidence || retry?.payload.summary || session.pause_reason || session.failure_reason || "模型调用未正常完成。");
  const advice = String(payload.advice || "稍后发送“继续”重试；若同一任务反复失败，请切换中转站或模型做对照。");
  const estimated = Number(context?.payload.estimated_tokens || retry?.payload.estimated_tokens || 0);
  const limit = Number(context?.payload.limit_tokens || 0);
  const contextUsage = estimated && limit ? `${estimated.toLocaleString()} / ${limit.toLocaleString()} Token（${Math.round(estimated / limit * 100)}%）` : estimated ? `约 ${estimated.toLocaleString()} Token` : "未取得";
  const latency = Number(payload.latency_ms || 0);
  const status = payload.status_code ? `HTTP ${String(payload.status_code)}` : String(payload.error_type || "调用中断");
  return <article className="runtime-diagnostic" aria-live="polite">
    <header><div><strong>任务已安全暂停</strong><small>代码与执行进度已经保存，不是项目文件突然丢失</small></div><b>{source} · {confidence}置信度</b></header>
    <div className="runtime-diagnostic-grid">
      <section><small>发生在哪里</small><strong>{source}</strong><p>{evidence}</p></section>
      <section><small>本次调用</small><strong>{status}</strong><p>{latency ? `等待 ${(latency / 1000).toFixed(1)} 秒后中断` : "连接未正常完成"}{retry ? "，已自动精简上下文并重试" : ""}。</p></section>
      <section><small>上下文状态</small><strong>{contextUsage}</strong><p>{context?.payload.trimmed ? "已自动裁剪较旧内容，保留当前任务证据。" : "当前上下文仍被保留，可从暂停位置恢复。"}</p></section>
      <section><small>你现在可以怎么做</small><strong>无需重新开始任务</strong><p>{advice}</p></section>
    </div>
  </article>;
}
function WorkingStatus({ session, events, now, runStartedAt }: { session: Session; events: Event[]; now: number; runStartedAt: number | null }) { const boundary = events.reduce((last, event, index) => event.event_type === "user_message" || event.event_type === "steer_applied" ? index : last, -1); const currentEvents = boundary >= 0 ? events.slice(boundary) : events; const latest = currentEvents.at(-1); const candidates = [runStartedAt, latest ? new Date(latest.created_at).getTime() : null, session.updated_at ? new Date(session.updated_at).getTime() : null].filter((value): value is number => value !== null && Number.isFinite(value) && value <= now); const started = candidates.length ? Math.max(...candidates) : now; const seconds = Math.max(0, Math.floor((now - started) / 1000)); const elapsed = seconds >= 60 ? `${Math.floor(seconds / 60)}分${seconds % 60}秒` : `${seconds}秒`; const phase = activityText[session.activity || "preparing_context"] || "处理中"; const detail = latest ? summary(latest) : "正在准备下一步操作。"; const waitingForModel = ["waiting_model", "retrying_model"].includes(session.activity || ""); const plan = currentEvents.some((event) => ["plan", "plan_resumed"].includes(event.event_type)) ? session.plan ?? [] : []; return <article className="agent-working" aria-live="polite"><header><span className="working-pulse"><i /><i /><i /></span><div><strong>{phase}</strong><small>本阶段已用时 {elapsed}{waitingForModel && seconds >= 90 ? " · 模型响应较慢，RAgent 仍在等待" : ""}</small></div><b>受控执行</b></header><p>{detail}</p>{plan.length > 0 && <ol className="working-plan">{plan.map((item) => <li className={item.status} key={item.key}><span><b>{item.title}</b>{item.note && <small>{item.note}</small>}</span></li>)}</ol>}</article>; }
function MessageContent({ content }: { content: string }) { if (content.startsWith("任务完成\n\n")) return <CompletionResult content={content} />; if (content.startsWith("调用诊断\n\n")) return <DiagnosticResult content={content} />; return <div className="message-content"><Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeHighlight, { detect: false }]]} components={{ pre: CodeBlock }}>{content}</Markdown></div>; }
function DiagnosticResult({ content }: { content: string }) { const sections = content.split(/\n{2,}/).slice(1).map((block) => { const [title, ...lines] = block.split("\n"); return { title, value: lines.join(" ").replace(/^\s*[-*]\s+/, "") }; }); const source = sections.find((item) => item.title === "判定来源"); return <div className="diagnostic-result"><header><div><strong>本次调用故障诊断</strong><small>{source?.value || "正在判断故障来源"}</small></div></header><div>{sections.filter((item) => item.title !== "判定来源").map((item) => <section key={item.title}><h4>{item.title}</h4><p>{item.value}</p></section>)}</div></div>; }
function CompletionResult({ content }: { content: string }) { const blocks = content.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean); const sectionNames = new Set(["完成内容", "修改文件", "验证结果", "Git 状态", "改动规模"]); const sections: Array<{ title: string; values: string[] }> = []; const notes: string[] = []; for (const block of blocks.slice(1)) { const [first, ...rest] = block.split("\n"); const title = first.trim(); const value = rest.join("\n").trim(); if (sectionNames.has(title) && value) sections.push({ title, values: value.split("\n").map((line) => line.replace(/^\s*[-*]\s+/, "")).filter(Boolean) }); else notes.push(block); } const verification = sections.find((section) => section.title === "验证结果")?.values.join(" ") || ""; const verified = !/(?:未运行|未验证|跳过)/.test(verification); return <div className="completion-result"><header><span>✓</span><div><strong>任务已完成</strong><small>{verified ? "结果与验证证据已保存" : "结果与执行状态已保存"}</small></div></header><div className="completion-grid">{sections.map((section) => <section className={`completion-${section.title}`} key={section.title}><h4>{section.title}</h4>{section.values.map((value) => <div className="completion-item" key={`${section.title}-${value}`}><i>{section.title === "验证结果" ? (verified ? "✓" : "–") : section.title === "修改文件" || section.title === "Git 状态" ? "⌘" : "·"}</i><span>{value}</span></div>)}</section>)}</div>{notes.length > 0 && <footer>{notes.join(" ")}</footer>}</div>; }
function SectionLabel({ children }: { children: React.ReactNode }) { return <div className="section-label">{children}</div>; }
function summary(event: Event) { const p = event.payload; if (event.event_type === "model_response" && Array.isArray(p.tool_names)) { const names = p.tool_names.join(" → "); const targets = Array.isArray(p.tool_targets) ? p.tool_targets.filter(Boolean).join("；") : ""; const latency = Number(p.latency_ms || 0); return `工具：${names}${targets ? ` · 目标：${targets}` : ""} · ${String(p.protocol || "unknown")}${latency > 0 ? ` · ${latency} ms` : ""}`; } return String(p.summary || p.rationale || p.reason || "状态已保存"); }
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
  const latestCompression = latest
    ? `${latest.event_type === "context_compressed" ? "压缩" : "裁剪"}${before && after ? ` ${before.toLocaleString()} → ${after.toLocaleString()}` : "已完成"}`
    : "";
  const percent = used ? Math.min(100, (used / limit) * 100) : null;
  const remaining = percent === null ? "—" : `剩余 ${Math.max(0, 100 - percent).toFixed(percent < 1 ? 2 : 1)}%`;
  const detail = used ? `${measured ? "实际输入" : "预估输入"} ${used.toLocaleString()} / ${limit.toLocaleString()} Token` : `尚无模型调用 · 上限 ${limit.toLocaleString()} Token`;
  const actualDetail = measured ? "服务商返回实际输入；压缩按本地估算触发" : running ? "实际用量待返回；压缩按本地估算触发" : "实际用量未提供；压缩按本地估算触发";
  return { percent, remaining, detail, actualDetail, compressions: compressions.length, latestCompression };
}
function projectName(path: string) { return path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || path; }

function buildFileTree(paths: string[], directories: string[] = []): FileNode[] {
  const root: FileNode = { name: "", path: "", kind: "folder", children: [] };
  const entries: Array<[string, "file" | "folder"]> = [...directories.map((path) => [path, "folder"] as [string, "folder"]), ...paths.map((path) => [path, "file"] as [string, "file"])];
  for (const [rawPath, leafKind] of entries) {
    const parts = rawPath.replace(/\\/g, "/").split("/").filter(Boolean);
    let parent = root;
    parts.forEach((name, index) => {
      const path = parts.slice(0, index + 1).join("/");
      const kind = index === parts.length - 1 ? leafKind : "folder";
      let node = parent.children.find((item) => item.name === name && item.kind === kind);
      if (!node) { node = { name, path, kind, children: [] }; parent.children.push(node); }
      parent = node;
    });
  }
  const sort = (nodes: FileNode[]): FileNode[] => nodes
    .sort((left, right) => left.kind === right.kind ? left.name.localeCompare(right.name) : left.kind === "folder" ? -1 : 1)
    .map((node) => ({ ...node, children: sort(node.children) }));
  return sort(root.children);
}

function FileTreeNode({ node, changed, onOpen, onSelectFolder, selectedFolder, depth = 0 }: { node: FileNode; changed: Set<string>; onOpen: (path: string) => void; onSelectFolder: (path: string) => void; selectedFolder: string; depth?: number }) {
  const [expanded, setExpanded] = useState(depth < 1);
  const hasChange = node.kind === "file" ? changed.has(node.path) : [...changed].some((path) => path === node.path || path.startsWith(`${node.path}/`));
  const style = { "--tree-indent": `${7 + depth * 14}px` } as CSSProperties;
  if (node.kind === "folder") return <div className={`file-folder${expanded ? " expanded" : ""}`}>
    <button className={`${hasChange ? "changed " : ""}${selectedFolder === node.path ? "selected" : ""}`} style={style} type="button" aria-expanded={expanded} onClick={() => { onSelectFolder(node.path); setExpanded((value) => !value); }}>
      <i className="tree-chevron">›</i><span><strong>{node.name}</strong></span>{hasChange && <em>●</em>}
    </button>
    {expanded && <div className="folder-children">{node.children.map((child) => <FileTreeNode changed={changed} depth={depth + 1} key={child.path} node={child} onOpen={onOpen} onSelectFolder={onSelectFolder} selectedFolder={selectedFolder} />)}</div>}
  </div>;
  return <button className={`file-entry${hasChange ? " changed" : ""}`} style={style} type="button" title={node.path} onClick={() => onOpen(node.path)}>
    <i className="tree-spacer" /><span><strong>{node.name}</strong></span>{hasChange && <em>已修改</em>}
  </button>;
}
function ComposerMeta({ session, onChanged }: { session: Session | null; onChanged: () => Promise<void> }) {
  const [models, setModels] = useState<Record<Provider, { selected: string; choices: string[] }> | null>(null);
  useEffect(() => { api.models().then(setModels).catch(() => setModels(null)); }, []);
  if (!session) return <footer><span>未选择会话</span></footer>;
  const current = session;
  const options = effort[current.provider];
  async function save(provider: Provider, model: string, reasoning = current.reasoning_effort) { if (current.status === "running") return; const allowed = effort[provider].map(([value]) => value); const nextReasoning = allowed.includes(reasoning) ? reasoning : (provider === "deepseek" ? "high" : "medium"); await api.updateSettings(current.session_id, { provider, model, reasoning_effort: nextReasoning, response_style: current.response_style, verification_mode: current.verification_mode, test_command: current.test_command }); await onChanged(); }
  const choices = models?.[current.provider]?.choices ?? [current.model];
  return <footer className="model-controls"><label>API<select value={current.provider} disabled={current.status === "running"} onChange={(e) => { const provider = e.target.value as Provider; save(provider, models?.[provider]?.selected || models?.[provider]?.choices[0] || ""); }}><option value="deepseek">DeepSeek</option><option value="openai">OpenAI 中转站</option><option value="openai_official">OpenAI 官方</option></select></label><label>模型<select value={current.model} disabled={current.status === "running"} onChange={(e) => save(current.provider, e.target.value)}>{choices.map((model) => <option key={model}>{model}</option>)}</select></label><EffortSlider current={current} options={options} onCommit={(selected) => save(current.provider, current.model, selected)} /></footer>;
}

function EffortSlider({ current, options, onCommit }: { current: Session; options: Array<[string, string]>; onCommit: (value: string) => Promise<void> }) {
  const serverIndex = Math.max(0, options.findIndex(([value]) => value === current.reasoning_effort));
  const [position, setPosition] = useState(() => Math.round(serverIndex / Math.max(1, options.length - 1) * 1000));
  const [dragging, setDragging] = useState(false);
  useEffect(() => { setPosition(Math.round(serverIndex / Math.max(1, options.length - 1) * 1000)); }, [current.session_id, current.provider, current.reasoning_effort, serverIndex, options.length]);
  function indexAt(value: number) { return Math.round(value / 1000 * (options.length - 1)); }
  async function commit(value: number) { const selected = options[indexAt(value)]?.[0]; if (selected && selected !== current.reasoning_effort) await onCommit(selected); }
  const level = options[indexAt(position)]?.[1] || "";
  return <label className={`effort-control${dragging ? " dragging" : ""}`} style={{ "--effort-progress": `${position / 10}%` } as CSSProperties}>
    <span className="meteor-rail"><span className="meteor-track"><span className="meteor-tail" /><span className="meteor-head" /></span><input type="range" min="0" max="1000" value={position} aria-label="推理强度" aria-valuetext={level} disabled={current.status === "running"} onChange={(e) => setPosition(Number(e.target.value))} onPointerDown={() => setDragging(true)} onPointerUp={(e) => { setDragging(false); commit(Number(e.currentTarget.value)); }} onPointerCancel={(e) => { setDragging(false); commit(Number(e.currentTarget.value)); }} onKeyUp={(e) => commit(Number(e.currentTarget.value))} onBlur={(e) => { setDragging(false); commit(Number(e.currentTarget.value)); }} /></span>
    <span className="effort-level">{level}</span>
  </label>;
}

function ModalCard({ children, onClose }: { children: React.ReactNode; onClose: () => void }) { return <div className="modal" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}><section className="modal-card">{children}</section></div>; }

function NewSession({ onClose, onCreated }: { onClose: () => void; onCreated: (id: string) => void }) {
  const [provider, setProvider] = useState<Provider>("deepseek"), [path, setPath] = useState(""), [model, setModel] = useState("deepseek-v4-flash"), [picker, setPicker] = useState(false), [error, setError] = useState("");
  useEffect(() => { Promise.all([api.providers(), api.models()]).then(([states, models]) => { const next: Provider = states.deepseek.configured ? "deepseek" : states.openai_official.configured ? "openai_official" : states.openai.configured ? "openai" : "deepseek"; setProvider(next); setModel(models[next].selected || models[next].choices[0]); }).catch((e) => setError(e.message)); }, []);
  async function submit(e: FormEvent<HTMLFormElement>) { e.preventDefault(); const data = new FormData(e.currentTarget); try { const result = await api.createSession({ repo_root: path, provider, model, reasoning_effort: provider === "deepseek" ? "high" : "medium", permission_mode: data.get("permission_mode"), verification_mode: data.get("verification_mode"), test_command: String(data.get("test_command") || "") }); onCreated(result.session_id); } catch (e) { setError((e as Error).message); } }
return <ModalCard onClose={onClose}><form onSubmit={submit}><ModalHead title="新建对话" onClose={onClose} /><p className="modal-intro">先选择项目。创建后可在输入框下方随时选择 API、模型和推理强度。</p><label>仓库文件夹<div className="path-row"><input value={path} onChange={(e) => setPath(e.target.value)} required /><button type="button" onClick={() => setPicker(true)}>选择</button></div></label><label>权限级别<select name="permission_mode" defaultValue="important"><option value="ask">逐项批准</option><option value="important">仅重要操作批准（推荐）</option><option value="full">完全批准</option></select></label><label>验证模式<select name="verification_mode"><option value="auto">自动验证</option><option value="quick">快速模式</option><option value="strict">严格验证</option></select></label><label>验证命令（可选）<input name="test_command" defaultValue="" placeholder="留空按项目自动选择；严格模式需填写" /></label>{error && <p className="form-error">{error}</p>}<div className="form-actions"><button type="button" onClick={onClose}>取消</button><button className="primary">创建对话</button></div></form>{picker && <FolderPicker onSelect={(value) => { setPath(value); setPicker(false); }} onClose={() => setPicker(false)} />}</ModalCard>;
}

function FolderPicker({ onSelect, onClose }: { onSelect: (path: string) => void; onClose: () => void }) {
  const [data, setData] = useState<{ current: string | null; parent: string | null; directories: Array<{ name: string; path: string }> } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);
  const [folderName, setFolderName] = useState("");
  async function load(path?: string) {
    setLoading(true);
    setError("");
    try {
      setData(await api.directories(path));
      setSelected(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { void load(); }, []);
  async function createFolder() {
    if (!data?.current || !folderName.trim()) return;
    setLoading(true);
    setError("");
    try {
      const created = await api.createDirectory(data.current, folderName.trim());
      await load(data.current);
      setSelected(created.path);
      setFolderName("");
      setCreating(false);
    } catch (e) {
      setError((e as Error).message);
      setLoading(false);
    }
  }
  const target = selected || data?.current;
  return <div className="folder-pop">
    <header><strong>选择文件夹</strong><nav><button type="button" disabled={!data?.current} onClick={() => setCreating((value) => !value)}>＋ 新建文件夹</button><button type="button" onClick={onClose}>×</button></nav></header>
    <div className="folder-location"><p title={data?.current || "此电脑"}>{data?.current || "此电脑"}</p>{creating && <div className="new-folder-row"><input autoFocus value={folderName} onChange={(e) => setFolderName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") void createFolder(); if (e.key === "Escape") setCreating(false); }} placeholder="输入文件夹名称" /><button type="button" disabled={!folderName.trim() || loading} onClick={() => void createFolder()}>创建</button></div>}</div>
    <div className="folder-list">
      <button type="button" onClick={() => void load(data?.parent || undefined)}>↰ {data?.parent ? "上一级" : "返回此电脑"}</button>
      {loading && <span className="folder-hint">正在读取文件夹…</span>}
      {!loading && data?.directories.map((directory) => <button type="button" className={selected === directory.path ? "selected" : ""} key={directory.path} onClick={() => setSelected(directory.path)} onDoubleClick={() => void load(directory.path)} title="单击选中，双击打开">📁 {directory.name}</button>)}
      {error && <span className="folder-error">{error}</span>}
    </div>
    <footer><small>{selected ? "已选中此文件夹；双击可继续进入" : "单击选择，双击打开文件夹"}</small><button type="button" className="primary" disabled={!target || loading} onClick={() => target && onSelect(target)}>选择此文件夹</button></footer>
  </div>;
}

function ModalHead({ title, onClose }: { title: string; onClose: () => void }) { return <header className="modal-head"><div><small>RAgent workspace</small><h2>{title}</h2></div><button type="button" onClick={onClose}>×</button></header>; }

function SessionSettings({ session, onClose, onSaved }: { session: Session; onClose: () => void; onSaved: (updated: Session) => void }) { const [saving, setSaving] = useState(false); const [saveError, setSaveError] = useState(""); async function submit(e: FormEvent<HTMLFormElement>) { e.preventDefault(); if (saving) return; const d = new FormData(e.currentTarget); setSaving(true); setSaveError(""); try { const updated = await api.updateSettings(session.session_id, { provider: session.provider, model: session.model, reasoning_effort: String(d.get("reasoning_effort") || session.reasoning_effort), response_style: String(d.get("response_style") || session.response_style), permission_mode: String(d.get("permission_mode") || session.permission_mode || "important"), verification_mode: String(d.get("verification_mode") || session.verification_mode), test_command: String(d.get("test_command") || "") }); onSaved(updated); } catch (e) { setSaveError(`保存失败：${(e as Error).message}`); } finally { setSaving(false); } } return <ModalCard onClose={onClose}><form onSubmit={submit}><ModalHead title="会话设置" onClose={onClose} /><label>模型<input value={`${session.provider} · ${session.model}`} disabled /></label><label>回答风格<select name="response_style" defaultValue={session.response_style}><option value="concise">精简</option><option value="standard">标准</option><option value="teaching">教学</option></select></label><label>推理强度<select name="reasoning_effort" defaultValue={session.reasoning_effort}>{effort[session.provider].map(([v, l]) => <option value={v} key={v}>{l}</option>)}</select></label><label>权限级别<select name="permission_mode" defaultValue={session.permission_mode || "important"}><option value="ask">逐项批准</option><option value="important">仅重要操作批准（推荐）</option><option value="full">完全批准</option></select></label><p>完全批准仍遵守任务限制与项目边界；程序可以执行有副作用的命令，请仅对可信项目启用。</p><label>验证模式<select name="verification_mode" defaultValue={session.verification_mode}><option value="auto">自动验证</option><option value="quick">快速模式</option><option value="strict">严格验证</option></select></label><label>验证命令（可选）<input name="test_command" defaultValue={session.test_command.join(" ")} placeholder="留空按项目自动选择" /></label>{saveError && <p className="form-error" role="alert">{saveError}</p>}<div className="form-actions"><button type="button" disabled={saving} onClick={onClose}>取消</button><button type="submit" disabled={saving} className="primary">{saving ? "保存中…" : "保存"}</button></div></form></ModalCard>; }

function UpgradeAcceptance({ onClose, onBack }: { onClose: () => void; onBack: () => void }) {
  const [version, setVersion] = useState("3.0.0");
  const [checks, setChecks] = useState<UpgradeCheck[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { api.upgradeChecks().then((result) => { setVersion(result.version); setChecks(result.checks); }).catch((e) => setError((e as Error).message)); }, []);
  async function runOne(id: string) {
    setRunning(id); setError("");
    setChecks((items) => items.map((item) => item.id === id ? { ...item, status: "running" } : item));
    try {
      const result = await api.runUpgradeCheck(id) as UpgradeCheck;
      setChecks((items) => items.map((item) => item.id === id ? result : item));
    } catch (e) { setError((e as Error).message); setChecks((items) => items.map((item) => item.id === id ? { ...item, status: "failed", error: (e as Error).message } : item)); }
    finally { setRunning(null); }
  }
  async function runAll() {
    setRunning("all"); setError("");
    setChecks((items) => items.map((item) => ({ ...item, status: "running" })));
    try {
      const result = await api.runUpgradeCheck("all") as { version: string; results: UpgradeCheck[] };
      setVersion(result.version); setChecks(result.results);
    } catch (e) { setError((e as Error).message); }
    finally { setRunning(null); }
  }
  const passed = checks.filter((item) => item.status === "passed").length;
  return <ModalCard onClose={onClose}><ModalHead title="升级验收" onClose={onClose} /><section className="acceptance-hero"><div><small>RAgent {version}</small><strong>{checks.length ? `${passed} / ${checks.length} 项通过` : "正在读取能力清单"}</strong><p>每项测试直接调用当前版本的生产代码，不消耗模型 Token。</p></div><button className="primary" disabled={Boolean(running) || !checks.length} onClick={() => void runAll()}>{running === "all" ? "正在全部验收…" : "全部验收"}</button></section><div className="acceptance-list">{checks.map((check) => <article className={`acceptance-check ${check.status}`} key={check.id}><div><header><strong>{check.title}</strong><span>{check.status === "passed" ? "已通过" : check.status === "failed" ? "未通过" : check.status === "running" ? "测试中" : "未测试"}</span></header><p>{check.description}</p>{check.evidence?.map((line) => <small key={line}>{line}</small>)}{check.error && <em>{check.error}</em>}</div><button disabled={Boolean(running)} onClick={() => void runOne(check.id)}>{running === check.id ? "测试中…" : check.status === "untested" ? "开始测试" : "重新测试"}</button></article>)}</div>{error && <p className="notice">{error}</p>}<div className="form-actions"><button onClick={onBack}>返回 API 配置</button><button className="primary" onClick={onClose}>完成</button></div></ModalCard>;
}

function ApiSettings({ onClose }: { onClose: () => void }) {
  const cachedQuota = useMemo(readQuotaCache, []);
  const [provider, setProvider] = useState<Provider>("deepseek"), [states, setStates] = useState<ProviderState | null>(null), [models, setModels] = useState<Record<Provider, { selected: string; choices: string[] }> | null>(null), [quotas, setQuotas] = useState<Partial<Record<Provider, Quota>>>(cachedQuota?.quotas || {}), [key, setKey] = useState(""), [base, setBase] = useState("https://api.openai.com/v1"), [selected, setSelected] = useState(""), [notice, setNotice] = useState("");
  const [quotaToken, setQuotaToken] = useState(""), [quotaRefreshToken, setQuotaRefreshToken] = useState(""), [quotaUserId, setQuotaUserId] = useState(""), [refreshing, setRefreshing] = useState(false), [checkedAt, setCheckedAt] = useState(() => cachedQuota ? formatQuotaTime(cachedQuota.checkedAt) : ""), [testing, setTesting] = useState(false), [testResult, setTestResult] = useState<{ ok: boolean; text: string } | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(() => localStorage.getItem("ragent_quota_auto_refresh") !== "off");
  async function refreshQuotas() { setRefreshing(true); try { const results = await Promise.allSettled([api.quota("deepseek"), api.quota("openai"), api.quota("openai_official")]); const next = { deepseek: results[0].status === "fulfilled" ? results[0].value : quotas.deepseek, openai: results[1].status === "fulfilled" ? results[1].value : quotas.openai, openai_official: results[2].status === "fulfilled" ? results[2].value : quotas.openai_official }; const now = Date.now(); setQuotas(next); setCheckedAt(formatQuotaTime(now)); localStorage.setItem("ragent_quota_cache", JSON.stringify({ checkedAt: now, quotas: next })); } finally { setRefreshing(false); } }
  async function load() { const [s, m] = await Promise.all([api.providers(), api.models()]); setStates(s); setModels(m); setBase(s.openai.base_url || "https://api.openai.com/v1"); setQuotaUserId(s.openai.quota_user_id || ""); setSelected(m[provider].selected); }
  useEffect(() => { load().catch((e) => setNotice(e.message)); }, [provider]);
  useEffect(() => { setKey(""); setTestResult(null); }, [provider]);
  useEffect(() => { if (!cachedQuota || Date.now() - cachedQuota.checkedAt > 2 * 60 * 1000) void refreshQuotas(); }, []);
  useEffect(() => { localStorage.setItem("ragent_quota_auto_refresh", autoRefresh ? "on" : "off"); if (!autoRefresh) return; const timer = window.setInterval(() => { void refreshQuotas(); }, 5 * 60 * 1000); return () => window.clearInterval(timer); }, [autoRefresh]);
  async function discover() { try { if (provider === "deepseek") return; const result = await api.discoverModels(provider, provider === "openai_official" ? "https://api.openai.com/v1" : base, key); setModels((m) => m ? { ...m, [provider]: { selected: result.models.includes(selected) ? selected : result.models[0], choices: result.models } } : m); setSelected((current) => result.models.includes(current) ? current : result.models[0]); setNotice(`读取到 ${result.models.length} 个模型`); } catch (e) { setNotice((e as Error).message); } }
  async function save() { try { if (key) await api.saveKey(provider, key); if (provider === "openai") await api.saveOpenAI(base, selected, quotaToken, quotaRefreshToken, quotaUserId); else await api.saveModel(provider, selected); setNotice("配置已保存"); setKey(""); setQuotaToken(""); setQuotaRefreshToken(""); await load(); await refreshQuotas(); } catch (e) { setNotice((e as Error).message); } }
  async function testConnection() { setTesting(true); setTestResult(null); try { const result = await api.testConnection(provider, selected, provider === "openai" ? base : undefined, key); const status = result.status_code ? `HTTP ${result.status_code} · ` : ""; setTestResult({ ok: result.ok, text: `${status}${result.latency_ms} ms · ${result.message}` }); } catch (e) { setTestResult({ ok: false, text: (e as Error).message }); } finally { setTesting(false); } }
  const choices = models?.[provider]?.choices || [];
  return <ModalCard onClose={onClose}><ModalHead title="API 配置" onClose={onClose} /><section className="quota-panel"><header><div><strong>额度概览</strong><small>{checkedAt ? `最近刷新 ${checkedAt}` : "打开设置不会立即请求额度"}</small></div><nav className="quota-actions"><label className="quota-auto"><input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} /><span>每 5 分钟</span></label><button className={`quota-refresh ${refreshing ? "refreshing" : ""}`} disabled={refreshing} onClick={() => refreshQuotas()}><i>↻</i><span>{refreshing ? "读取中" : "刷新"}</span></button></nav></header><div>{(["deepseek", "openai", "openai_official"] as Provider[]).map((name) => <article key={name}><span>{name === "deepseek" ? "DeepSeek" : name === "openai" ? "OpenAI / 中转站" : "OpenAI 官方"}</span><strong>{quotaText(quotas[name])}</strong><small>{name === "openai_official" ? "官方 API Key 不提供剩余额度查询；请在 OpenAI 平台查看用量和费用。" : quotaHint(quotas[name])}</small></article>)}</div></section><div className="provider-tabs"><button className={provider === "deepseek" ? "active" : ""} onClick={() => { setProvider("deepseek"); setTestResult(null); }}>DeepSeek</button><button className={provider === "openai" ? "active" : ""} onClick={() => { setProvider("openai"); setTestResult(null); }}>中转站</button><button className={provider === "openai_official" ? "active" : ""} onClick={() => { setProvider("openai_official"); setTestResult(null); }}>OpenAI 官方</button></div><p className="credential-state">{states?.[provider]?.configured ? "● 已配置" : "○ 尚未配置"}</p><label>API Key<input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder={states?.[provider]?.configured ? "输入新密钥可更新" : "粘贴 API Key"} /></label>{provider === "openai" && <><label>兼容 Base URL<div className="path-row"><input value={base} onChange={(e) => setBase(e.target.value)} /><button onClick={discover}>读取模型</button></div></label><section className="quota-credentials"><header><div><strong>Sub2API 账户额度</strong><small>令牌过期后自动续期并继续查询</small></div><span>{states?.openai.quota_refresh_configured ? "自动续期" : states?.openai.quota_configured ? "已配置" : "可选"}</span></header><label>auth_token<input type="password" value={quotaToken} onChange={(e) => setQuotaToken(e.target.value)} placeholder={states?.openai.quota_configured ? "输入新令牌可更新" : "localStorage 中的 auth_token"} /></label><label>refresh_token<input type="password" value={quotaRefreshToken} onChange={(e) => setQuotaRefreshToken(e.target.value)} placeholder={states?.openai.quota_refresh_configured ? "已加密保存，输入可更新" : "localStorage 中的 refresh_token"} /></label><label>NewAPI 用户 ID（Sub2API 留空）<input value={quotaUserId} onChange={(e) => setQuotaUserId(e.target.value)} placeholder="仅 NewAPI 需要，例如 1" /></label></section></>}<label>默认模型<select value={selected} onChange={(e) => { setSelected(e.target.value); setTestResult(null); }}>{choices.map((m) => <option key={m}>{m}</option>)}</select></label>{provider === "openai_official" && <button type="button" onClick={discover}>读取官方模型</button>}<section className={`connection-test ${testResult ? (testResult.ok ? "success" : "failure") : ""}`}><div><strong>连接测试</strong><small>使用当前密钥、地址和模型发送一个最小请求</small></div><button type="button" disabled={testing || !selected} onClick={() => void testConnection()}>{testing ? "测试中…" : "测试是否可用"}</button>{testResult && <p>{testResult.text}</p>}</section>{notice && <p className="notice">{notice}</p>}<div className="form-actions"><button onClick={onClose}>关闭</button><button className="primary" onClick={save}>保存配置</button></div></ModalCard>;
}

function quotaText(quota?: Quota) { if (!quota) return "未查询"; if (!quota.supported) return "平台查看"; if (!quota.configured) return "未配置"; if (quota.error?.includes("未提供")) return "未开放余额接口"; if (quota.error?.includes("后台访问令牌")) return "需要后台令牌"; if (quota.error || quota.is_available === false) return "查询失败"; const balance = quota.balances[0]; return balance ? `${balance.currency} ${balance.total_balance}` : "可用"; }
function quotaHint(quota?: Quota) { if (!quota) return "正在后台读取额度"; if (quota.error?.includes("未提供")) return "已尝试 NewAPI、Sub2API 与标准余额端点"; return quota.error || (quota.supported ? "最近一次真实额度状态" : "请前往供应商平台查看"); }
function formatQuotaTime(value: number) { return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
function readQuotaCache(): { checkedAt: number; quotas: Partial<Record<Provider, Quota>> } | null { try { const value = JSON.parse(localStorage.getItem("ragent_quota_cache") || "null"); return value && typeof value.checkedAt === "number" && value.quotas ? value : null; } catch { return null; } }

function FileDrawer({ value, onClose }: { value: NonNullable<FileView>; onClose: () => void }) {
  const [view, setView] = useState<"diff" | "before" | "after">(value.changed ? "diff" : "after");
  const content = view === "before" ? value.before || "没有修改前快照" : value.after;
  return <div className="drawer"><header><div><strong>{value.path}</strong><small>{value.changed ? "已修改 · 绿色为新增，红色为删除" : "未修改"}</small></div><nav><button className={view === "before" ? "active" : ""} onClick={() => setView("before")}>更改前</button><button className={view === "after" ? "active" : ""} onClick={() => setView("after")}>更改后</button><button className={view === "diff" ? "active" : ""} onClick={() => setView("diff")}>差异对比</button><button className="drawer-close" onClick={onClose}>×</button></nav></header>{view === "diff" ? <DiffViewer diff={value.diff} /> : <pre className="source-view">{content}</pre>}</div>;
}

function GitPanel({ session, onClose }: { session: Session; onClose: () => void }) {
  const [data, setData] = useState<GitPanelData | null>(null);
  const [selectedPath, setSelectedPath] = useState("");
  const [diff, setDiff] = useState("");
  const [branch, setBranch] = useState("");
  const [initialBranch, setInitialBranch] = useState("main");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const load = useCallback(async () => { const value = await api.git(session.session_id); setData(value); setNotice(value.initialized && !value.has_commits ? "当前仓库尚无提交；首次提交后分支将正式建立。" : ""); }, [session.session_id]);
  useEffect(() => { load().catch((e) => setNotice(e.message)); }, [load]);
  async function showDiff(path: string) { setSelectedPath(path); try { setDiff((await api.git(session.session_id, path)).diff ?? ""); } catch (e) { setNotice((e as Error).message); } }
  async function act(body: object, success: string) { const request = body as { action?: string; paths?: string[] }; const action = request.action; if (data?.has_commits === false && (action === "create_branch" || action === "switch_branch")) { setNotice("请先完成首次提交，再创建或切换分支。"); return; } setBusy(true); setNotice(""); try { const result = await api.gitAction(session.session_id, body); setBranch(""); setMessage(""); await load(); const hash = typeof result.commit === "string" ? result.commit.slice(0, 12) : ""; setNotice(action === "commit" ? `提交成功 · ${hash} · ${request.paths?.length || 0} 个文件` : success); } catch (e) { setNotice((e as Error).message); } finally { setBusy(false); } }
  const eligible = new Set(data?.commit_eligible || []);
  const commitPaths = (data?.status?.changes || []).map((line) => line.slice(3).split(" -> ").pop() || "").filter((path) => eligible.has(path));
  if (data && !data.initialized) return <div className="modal git-modal"><section className="git-card git-init-card"><header><div><small>Git 工作区</small><h2>初始化仓库</h2></div><nav><button onClick={onClose}>×</button></nav></header><main className="git-init"><h3>这个文件夹还不是 Git 仓库</h3><p>{data.repo_root}</p><p>初始化只会创建 <code>.git</code> 元数据，不会自动提交、删除或修改现有文件。</p><label>默认分支名<input value={initialBranch} onChange={(e) => setInitialBranch(e.target.value)} placeholder="main" /></label><button className="primary" disabled={busy || !initialBranch.trim()} onClick={() => act({ action: "initialize", branch: initialBranch.trim() }, "Git 仓库初始化成功")}>初始化 Git 仓库</button>{notice && <p className="notice">{notice}</p>}</main></section></div>;
  return <div className="modal git-modal"><section className="git-card"><header><div><small>Git 工作区</small><h2>{data?.branches?.current || data?.status?.branch || "读取中…"}</h2></div><nav><button disabled={busy} onClick={() => load().catch((e) => setNotice(e.message))}>↻ 刷新</button><button onClick={onClose}>×</button></nav></header><div className="git-grid"><section className="git-changes"><h3>工作区更改 <b>{data?.status?.changes.length || 0}</b></h3><div>{data?.status?.changes.map((line) => { const path = line.slice(3).split(" -> ").pop() || line; return <button className={selectedPath === path ? "active" : ""} key={line} onClick={() => showDiff(path)}><i>{line.slice(0, 2)}</i><span>{path}</span>{eligible.has(path) && <em>可提交</em>}</button>; })}<p className="git-empty">{data?.status?.changes.length ? "" : "工作区干净"}</p></div><label>提交说明<input value={message} onChange={(e) => setMessage(e.target.value)} placeholder="feat: 描述本次修改" /></label><button className="primary git-commit" disabled={busy || !message.trim() || !commitPaths.length} onClick={() => act({ action: "commit", message, paths: commitPaths }, "提交成功")}>提交 RAgent 本轮修改 ({commitPaths.length})</button></section><section className="git-diff"><h3>{selectedPath || "选择文件查看 Diff"}</h3><DiffViewer diff={diff} /></section><aside className="git-meta"><h3>分支</h3><select value={data?.branches?.current || ""} onChange={(e) => act({ action: "switch_branch", branch: e.target.value }, `已切换到 ${e.target.value}`)} disabled={busy}>{data?.branches?.branches.map((name) => <option key={name}>{name}</option>)}</select><div className="git-new-branch"><input value={branch} onChange={(e) => setBranch(e.target.value)} placeholder="feature/name" /><button disabled={busy || !branch.trim()} onClick={() => act({ action: "create_branch", branch }, `已创建并切换到 ${branch}`)}>创建</button></div><h3>最近提交</h3><div className="git-log">{data?.commits?.map((item) => <article key={item.sha}><code>{item.sha}</code><strong>{item.subject}</strong><small>{item.author} · {new Date(item.date).toLocaleDateString()}</small></article>)}</div></aside></div>{notice && <footer className="git-notice">{notice}</footer>}</section></div>;
}

function DiffViewer({ diff }: { diff: string }) {
  if (!diff.trim()) return <div className="diff-empty">当前文件没有可显示的差异记录</div>;
  let oldLine = 0; let newLine = 0;
  return <div className="diff-view" role="table" aria-label="代码差异">{diff.split("\n").map((line, index) => {
    let kind = "context"; let oldLabel = ""; let newLabel = "";
    const hunk = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (hunk) { oldLine = Number(hunk[1]); newLine = Number(hunk[2]); kind = "hunk"; }
    else if (line.startsWith("+++ ") || line.startsWith("--- ")) kind = "file";
    else if (line.startsWith("+")) { kind = "added"; newLabel = String(newLine++); }
    else if (line.startsWith("-")) { kind = "removed"; oldLabel = String(oldLine++); }
    else { oldLabel = String(oldLine++); newLabel = String(newLine++); }
    return <div className={`diff-line ${kind}`} role="row" key={`${index}-${line}`}><span className="line-number">{oldLabel}</span><span className="line-number">{newLabel}</span><code>{line || " "}</code></div>;
  })}</div>;
}

declare global { interface Window { pywebview?: { api?: { minimize?: () => void; maximize?: () => void; close?: () => void } } } }
