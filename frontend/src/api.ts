export type Provider = "deepseek" | "openai" | "openai_official";

export type Message = { role: "user" | "assistant"; content: string };
export type Session = {
  session_id: string;
  title: string;
  repo_root: string;
  provider: Provider;
  model: string;
  reasoning_effort: string;
  response_style: "concise" | "standard" | "teaching";
  permission_mode: "ask" | "important" | "full";
  verification_mode: "quick" | "auto" | "strict";
  test_command: string[];
  status: "idle" | "running" | "waiting_permission" | "paused" | "completed" | "failed";
  activity?: "idle" | "preparing_context" | "waiting_model" | "retrying_model" | "executing_tool" | "waiting_permission" | "paused" | "completed" | "failed";
  pause_reason?: string | null;
  step: number;
  turn_budget?: number;
  context_estimated_tokens?: number;
  context_actual_input_tokens?: number | null;
  context_limit_tokens?: number;
  context_trimmed_items?: string[];
  plan?: Array<{ key: string; title: string; status: "pending" | "in_progress" | "completed" | "blocked"; note?: string | null }>;
  task_contract?: { objective: string; intent: string; intent_confidence?: string; intent_rationale?: string | null; intent_source?: string; requires_clarification?: boolean; allowed_actions?: string[]; requirements: Array<{ key: string; description: string; expected?: string | null; satisfied: boolean; evidence?: string | null }> } | null;
  max_steps?: number;
  messages: Message[];
  changed_files: string[];
  failure_reason?: string | null;
  approved_paths: string[];
  pending_permission?: { decision?: Record<string, unknown> | null; request_id: string; path: string; reason: string; access: "read" | "write" | "execute" | "action"; command: string[]; follow_up_command?: string[]; capability?: string | null; operation?: string | null; purpose?: string | null; impact?: string | null; scope?: string | null; recovery?: string | null; recommendation?: string | null; destructive?: boolean; risk?: "low" | "medium" | "high" | "unknown" } | null;
  updated_at?: string;
};

export type Event = {
  sequence: number;
  event_type: string;
  created_at: string;
  payload: Record<string, unknown>;
};

export type ProviderState = Record<Provider, {
  configured: boolean;
  source: string;
  model?: string;
  base_url?: string;
  quota_configured?: boolean;
  quota_refresh_configured?: boolean;
  quota_user_id?: string;
}>;

export type Quota = {
  provider: Provider;
  supported: boolean;
  configured: boolean;
  is_available: boolean | null;
  low_balance: boolean | null;
  balances: Array<{ currency: string; total_balance: string }>;
  checked_at: string;
  error: string | null;
};

export type ConnectionTest = { ok: boolean; transport_ok?: boolean; protocol_ok?: boolean; provider: Provider; model: string; latency_ms: number; status_code: number | null; message: string };
export type UpgradeCheck = { id: string; title: string; description: string; status: "untested" | "running" | "passed" | "failed"; duration_ms?: number; evidence?: string[]; error?: string | null };
export type GitPanelData = {
  initialized: boolean;
  repo_root?: string;
  has_commits?: boolean;
  status?: { branch: string; changes: string[] };
  branches?: { branches: string[]; current: string };
  commits?: Array<{ sha: string; date: string; author: string; subject: string }>;
  diff?: string;
  commit_eligible?: string[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : `请求失败（HTTP ${response.status}）`;
    throw new Error(detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

export const api = {
  sessions: () => request<Session[]>("/studio-api/sessions"),
  session: (id: string) => request<Session>(`/studio-api/sessions/${id}`),
  events: (id: string) => request<Event[]>(`/studio-api/sessions/${id}/events`),
  files: (id: string) => request<{ files: string[]; directories: string[] }>(`/studio-api/sessions/${id}/files`),
  createProjectEntry: (id: string, kind: "file" | "folder", path: string) => request<{ kind: string; path: string }>(`/studio-api/sessions/${id}/files`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ kind, path }) }),
  git: (id: string, path?: string) => request<GitPanelData>(`/studio-api/sessions/${id}/git${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  gitAction: (id: string, body: object) => request<Record<string, unknown>>(`/studio-api/sessions/${id}/git`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
  fileChange: (id: string, path: string) => request<{ path: string; changed: boolean; before: string | null; after: string; diff: string }>(`/studio-api/sessions/${id}/file-change?path=${encodeURIComponent(path)}`),
  createSession: (body: object) => request<{ session_id: string }>("/studio-api/sessions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
  deleteSession: (id: string) => request<void>(`/studio-api/sessions/${id}`, { method: "DELETE" }),
  send: (id: string, content: string) => request<{ status: string }>(`/studio-api/sessions/${id}/messages`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ content }) }),
  updateSettings: (id: string, body: object) => request<Session>(`/studio-api/sessions/${id}/settings`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
  decidePermission: (id: string, requestId: string, approved: boolean, instruction?: string, scope = "once") => request<{ status: string }>(`/studio-api/sessions/${id}/permissions/${requestId}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ approved, scope, instruction: instruction || null }) }),
  providers: () => request<ProviderState>("/providers"),
  quota: (provider: Provider) => request<Quota>(`/providers/${provider}/quota`),
  models: () => request<Record<Provider, { selected: string; choices: string[] }>>("/provider-models"),
  saveKey: (provider: Provider, api_key: string) => request(`/providers/${provider}/credentials`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ api_key }) }),
  saveModel: (provider: Provider, model: string) => request(`/providers/${provider}/model`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ model }) }),
  saveOpenAI: (base_url: string, model: string, quota_access_token?: string, quota_refresh_token?: string, quota_user_id?: string) => request("/providers/openai/connection", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ base_url, model, quota_access_token: quota_access_token || null, quota_refresh_token: quota_refresh_token || null, quota_user_id: quota_user_id || null }) }),
  discoverModels: (provider: "openai" | "openai_official", base_url: string, api_key?: string) => request<{ models: string[] }>(`/providers/${provider}/models/discover`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ base_url, api_key: api_key || null }) }),
  testConnection: (provider: Provider, model: string, base_url?: string, api_key?: string) => request<ConnectionTest>(`/providers/${provider}/connection-test`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model, base_url: base_url || null, api_key: api_key || null }) }),
  upgradeChecks: () => request<{ version: string; checks: UpgradeCheck[] }>("/studio-api/upgrade-checks"),
  runUpgradeCheck: (id: string) => request<UpgradeCheck | { version: string; results: UpgradeCheck[] }>(`/studio-api/upgrade-checks/${id}`, { method: "POST" }),
  directories: (path?: string) => request<{ current: string | null; parent: string | null; directories: Array<{ name: string; path: string }> }>(`/system/directories${path ? `?path=${encodeURIComponent(path)}` : ""}`, { headers: { "x-veripatch-ui": "1" } }),
  createDirectory: (parent: string, name: string) => request<{ name: string; path: string }>("/system/directories", { method: "POST", headers: { "content-type": "application/json", "x-veripatch-ui": "1" }, body: JSON.stringify({ parent, name }) }),
};
