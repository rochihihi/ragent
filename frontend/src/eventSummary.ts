import type { Event } from "./api";

/** Display actual event content; raw payload remains available in the expanded entry. */
export function eventSummary(event: Event): string {
  const p = event.payload;
  if (event.event_type === "assistant_message") {
    return String(p.full_content || p.content || p.summary || "回答内容为空");
  }
  if (event.event_type === "model_response" && Array.isArray(p.tool_names)) {
    const names = p.tool_names.join(" → ");
    const targets = Array.isArray(p.tool_targets) ? p.tool_targets.filter(Boolean).join("；") : "";
    const latency = Number(p.latency_ms || 0);
    return `工具：${names}${targets ? ` · 目标：${targets}` : ""} · ${String(p.protocol || "unknown")}${latency > 0 ? ` · ${latency} ms` : ""}`;
  }
  if (event.event_type === "claim_review" && Array.isArray(p.claims)) {
    const unmatched = p.claims.filter((claim: { source_matched?: boolean }) => claim.source_matched !== true).length;
    return `审核 ${p.claims.length} 条引用，${unmatched} 条未匹配证据；未替换模型回答。`;
  }
  const target = p.path || (Array.isArray(p.command) ? p.command.join(" ") : "") || p.query;
  const description = String(p.summary || p.rationale || p.reason || p.content || p.reviewed_message || "状态已保存");
  return target ? `${description}\n目标：${String(target)}` : description;
}
