import { useEffect, useState } from "react";

type Skill = { name: string; description: string; content: string };
export function Skills({ id, onClose }: { id: string; onClose: () => void }) {
  const [items, setItems] = useState<Skill[]>([]);
  const [enabled, setEnabled] = useState<string[]>([]);
  const [saved, setSaved] = useState<string[]>([]);
  const [content, setContent] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const url = `/studio-api/sessions/${encodeURIComponent(id)}/skills`;
  async function request(method = "GET", body?: object) {
    const response = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "技能请求失败");
    return data;
  }
  async function load(preserveSelection = false) { const data = await request(); setItems(data.items); setSaved(data.enabled); if (!preserveSelection) setEnabled(data.enabled); }
  const dirty = [...enabled].sort().join("|") !== [...saved].sort().join("|");
  function close() { if (dirty) { setError("启用项尚未保存，请点击“保存启用项”，或恢复原勾选后关闭。"); return; } onClose(); }
  useEffect(() => { load().catch(e => setError(String(e.message))); }, [id]);
  async function perform(action: () => Promise<void>) {
    setBusy(true); setError("");
    try { await action(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  return <div className="modal"><section className="modal-card">
    <h2>项目技能</h2><p>导入 SKILL.md 后，勾选本会话启用。仅导入说明文件，不执行脚本。请只使用可信内容。</p>
    <p>存放位置：项目的 .agents/skills/技能名/SKILL.md</p>
    <p>已保存启用：{saved.length ? saved.join("、") : "无（导入不等于启用）"}{dirty ? "；当前勾选尚未保存" : ""}</p>
    {items.length === 0 && <p>还没有技能。可以导入文件或粘贴完整内容。</p>}
    {items.map(item => <div key={item.name}><label><input type="checkbox" disabled={busy} checked={enabled.includes(item.name)} onChange={e => setEnabled(e.target.checked ? [...enabled, item.name] : enabled.filter(n => n !== item.name))} />{item.name}：{item.description}</label><details><summary>查看说明</summary><pre style={{ whiteSpace: "pre-wrap" }}>{item.content}</pre></details></div>)}
    <label>选择 SKILL.md<input type="file" accept=".md" disabled={busy} onChange={async e => {
      const file = e.target.files?.[0]; if (!file) return;
      if (file.size > 48000) { setError("文件过大，请使用不超过 12000 字符的技能"); return; }
      setContent(await file.text());
    }} /></label>
    <label>技能内容<textarea rows={9} value={content} maxLength={12000} onChange={e => setContent(e.target.value)} placeholder={'---\nname: my-skill\ndescription: 适用场景\n---\n具体工作流程'} /></label>
    {error && <p role="alert" className="form-error">{error}</p>}
    <div className="form-actions"><button disabled={busy} onClick={close}>关闭</button><button disabled={busy || !content.trim()} onClick={() => perform(async () => { await request("POST", { content }); setContent(""); await load(true); })}>导入技能</button><button className="primary" disabled={busy} onClick={() => perform(async () => { const result = await request("PUT", { names: enabled }); setSaved(result.enabled); onClose(); })}>保存启用项</button></div>
  </section></div>;
}
