export type ApprovalAction = {
  number: string;
  action: string;
  rationale: string;
  command: string[];
  raw: Record<string, unknown>;
};

// Preserve every leaf and argument; no semantic classification or truncation.
export function approvalActions(decision: Record<string, unknown> | null | undefined): ApprovalAction[] {
  if (!decision) return [];
  function visit(item: Record<string, unknown>, number: string): ApprovalAction[] {
    if (item.action === "batch" && Array.isArray(item.actions) && item.actions.length) {
      return item.actions.flatMap((child, index) =>
        child && typeof child === "object" && !Array.isArray(child)
          ? visit(child as Record<string, unknown>, number ? `${number}.${index + 1}` : String(index + 1))
          : [{ number: String(index + 1), action: "unknown", rationale: "",
               command: [], raw: { invalid_action: child } }]
      );
    }
    return [{
      number: number || "1",
      action: typeof item.action === "string" ? item.action : "unknown",
      rationale: typeof item.rationale === "string" ? item.rationale : "",
      command: Array.isArray(item.command) && item.command.every(arg => typeof arg === "string")
        ? item.command as string[] : [],
      raw: item,
    }];
  }
  return visit(decision, "");
}
