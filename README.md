# RAgent

A local-first Windows desktop coding agent: the model chooses actions; the runtime controls execution, permissions, and recovery.

[中文说明](README.zh-CN.md)

![RAgent desktop workspace with external MCP tools](assets/ragent-studio-mcp-20261005.png)

Connect DeepSeek, OpenAI, or an OpenAI-compatible gateway to your local project. Ask RAgent to inspect code, create files, implement changes, run checks, or explain results. Conversations and task evidence are stored locally in SQLite.

Local-first does not mean offline inference: selected project content is sent to your configured provider. External MCP tools may also receive arguments.

## Features

- **Model-led workflow:** one Studio runtime. The model interprets the original request and chooses tools, plans, verification, and responses. Plans do not grant permissions.
- **Desktop workspace:** persistent conversations, project tree, file creation, Diff review, Git operations, skills, execution history, and per-request context usage.
- **Skill packages:** folder/ZIP imports with scripts and references, safe YAML parsing, automatic/pinned/disabled session modes, `$name` invocation, and three-tier loading. Updates, removal, and legacy Windows permission repairs preserve backups. [Skill guide](docs/skills.md).
- **Controlled execution:** concrete command/argument previews, one-time and matching session approvals, native workspace path checks, and protected repository metadata.
- **Pause/resume:** no default fixed decision-step limit. Pending model I/O can be cancelled; synchronous tools finish and save results before pausing. Ambiguous recovered calls are not blindly replayed.
- **Verification:** the model chooses relevant checks and may write tests. Failures return as tool evidence. Default modes do not enforce a test-and-reopen sequence; Strict adds completion gates. Completion is not proof of passing tests or visual GUI inspection.
- **Context management:** model-generated history summaries with coverage tracking, local budgeting, and fallback trimming. Current context size is separate from cumulative token usage.
  Official OpenAI Responses also enable server-side automatic compaction (default 200,000-token threshold, configurable with `VERIPATCH_OPENAI_COMPACT_THRESHOLD`). Tool-call chains retain compacted state through `previous_response_id`. New user turns still rebuild local context and retain history summaries; DeepSeek and gateways do not receive this parameter. Explicit unsupported-parameter rejections fall back to ordinary Responses; other request errors are not replayed for compaction fallback.
- **File protection:** creation conflicts do not silently become edits. Prepared edits re-match unique exact targets in current content, preserving unrelated changes; missing or ambiguous targets fail.
- **External MCP:** UI configuration for stdio and Streamable HTTP, connection testing, tool discovery, and approved calls. Local operations use native tools; no bundled MCP server EXE.
- **Credentials:** saved keys take priority over environment variables. System keyring storage has a Windows DPAPI-encrypted fallback. Deleting a saved key may reactivate an environment key.

## Execution model

```text
User request → context → model decision → permission/tool checks → execution
                   ↑                                      ↓
                   └──────── results and evidence ─────────┘
```

Python executes model-proposed actions, records observations, and returns evidence to the model. Responses are model-generated; auditing and mode-specific completion checks are local.

Multi-file rollback is best-effort: it restores text saved before the current application attempt only if the file still matches the content just written. There is no cross-process file lock, semantic merge, or guaranteed all-or-nothing recovery.

## Run from source

Requirements: Windows, Python 3.11+, Node.js/npm, a model-provider account, and a supported PyWebView backend (Windows may require WebView2 Runtime).

```powershell
git clone https://github.com/rochihihi/RAgent.git
cd RAgent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,desktop]"
npm --prefix frontend ci
npm --prefix sandbox/runtime ci --ignore-scripts --registry=https://registry.npmjs.org
npm --prefix frontend run build
.\.venv\Scripts\ragent-desktop.exe
```

Configure a provider in the app, choose a project folder, and create a conversation. Source execution uses editable installation so the desktop can locate the repository's built frontend.

The old `ragent`/`veripatch` CLI, including `serve`, `auth`, and `quota`, is removed. Desktop launchers remain; credentials and balance queries are managed in the GUI. Balance-query support depends on the provider/gateway.

## Build the Windows executable

Download the packaged Windows app from [GitHub Releases](https://github.com/rochihihi/RAgent/releases/latest). It still needs WebView2, a configured model provider, and setup of the experimental Windows sandbox. Python-based skill scripts need an external Python interpreter.

After installing the dependencies:

```powershell
.\.venv\Scripts\python.exe scripts\build_desktop.py
```

The script builds the frontend and generates a single `dist/RAgent.exe`. Executables, build caches, backups, and private configuration are not committed. Packaged use still needs provider network access; external stdio servers need their own runtime.

## External MCP

Configure servers in MCP service management. Configuration normally resides at `%APPDATA%\RAgent\mcp_servers.json`. Connection tests initialize the protocol and discover tools; they do not run every business operation. Calls use `server::tool`.

Only enable trusted servers: they may execute programs, modify data, or contact networks outside native file-tool safeguards. Approval is not a sandbox for third-party services.

Commands, tests, Git subprocesses and local stdio MCP now use a required-by-default experimental OS sandbox. Open Settings → Execution sandbox to check/setup the Windows alpha backend; no automatic elevation or unsandboxed fallback. Remote HTTP MCP effects remain outside this boundary. See [sandbox setup and limitations](docs/SANDBOX.md).

See [external MCP setup](docs/EXTERNAL_MCP.md). Current limitations: no OAuth flow, legacy HTTP SSE transport, or persistent connection reuse.

## Development checks

```powershell
.\.venv\Scripts\python.exe -m pytest
npm --prefix frontend run check:vue
npm --prefix frontend run test:trace
npm --prefix frontend run build
```

Tests cover model decisions, call identity, permissions, pause/resume, context summaries, file conflicts/rollback, persistence, and external MCP. Passing tests validate covered cases, not every provider or desktop environment.

## Layout and stack

```text
src/veripatch/   Studio runtime, adapters, tools, API, persistence
frontend/vue/   Vue 3 desktop interface
tests/          Unit and integration tests
scripts/        Desktop build script
assets/         Desktop resources and screenshot
examples/       Example projects
docs/           External MCP setup
```

Python, Pydantic, FastAPI/Uvicorn, SQLite, Vue 3/TypeScript/Vite, PyWebView, MCP SDK, and PyInstaller. The internal Python package is still named `veripatch`.

## Safety

Keep the local API on loopback; it is not a public execution service. Native file tools protect paths and metadata, but approved commands and external tools can have broader effects. Review destructive operations and use Git/backups for important work.

Sandbox approvals and execution isolation are separate. Desktop GUI and other incompatible operations can propose a per-call host exception, with fresh explicit approval rather than full/session grants; other calls remain sandboxed. Strict policy can forbid exceptions. Host calls have current-user access, not sandbox protection. Windows SRT is alpha, not a production-grade security guarantee.

Credentials are handled separately from model context, but files and tool output may contain secrets. Inspect what you expose to providers and external services.

## License

[MIT](LICENSE)
