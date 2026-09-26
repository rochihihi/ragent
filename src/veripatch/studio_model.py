"""Provider adapters for the interactive Studio action protocol."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from veripatch import __version__
from veripatch.config import Settings
from veripatch.credentials import load_api_key
from veripatch.studio_domain import (
    SemanticIntentAssessment,
    StudioAction,
    StudioClaim,
    StudioDecision,
    StudioReply,
)

_OPENAI_PROTOCOL_CACHE: dict[str, str] = {}


def _decision_tool() -> dict[str, Any]:
    """Return the single typed hand-off tool exposed to chat-compatible models."""
    return {
        "type": "function",
        "function": {
            "name": "submit_decision",
            "description": (
                "Submit the next audited RAgent action. Use this tool instead of "
                "describing an action in ordinary text."
            ),
            "parameters": StudioDecision.model_json_schema(),
        },
    }


_NATIVE_TOOL_FIELDS: dict[str, dict[str, dict[str, Any]]] = {
    "list_files": {},
    "search": {"query": {"type": "string"}},
    "read": {"path": {"type": "string"}},
    "edit": {
        "path": {"type": "string"},
        "old_text": {"type": "string"},
        "new_text": {"type": "string"},
    },
    "apply_patch": {"patch": {"type": "string"}},
    "create": {"path": {"type": "string"}, "content": {"type": "string"}},
    "move_file": {"path": {"type": "string"}, "destination": {"type": "string"}},
    "copy_file": {"path": {"type": "string"}, "destination": {"type": "string"}},
    "delete_path": {"path": {"type": "string"}},
    "git_status": {},
    "git_diff": {
        "path": {"type": ["string", "null"]},
        "revision": {"type": ["string", "null"]},
    },
    "git_log": {},
    "git_branch": {"branch": {"type": ["string", "null"]}},
    "git_commit": {"commit_message": {"type": "string"}},
    "git_restore": {
        "path": {"type": "string"},
        "revision": {"type": ["string", "null"]},
    },
    "run_tests": {"command": {"type": "array", "items": {"type": "string"}}},
    "run_command": {"command": {"type": "array", "items": {"type": "string"}}},
    "start_terminal": {"command": {"type": "array", "items": {"type": "string"}}},
    "poll_terminal": {"terminal_id": {"type": "string"}},
    "write_terminal": {"terminal_id": {"type": "string"}, "input": {"type": "string"}},
    "stop_terminal": {"terminal_id": {"type": "string"}},
    "inspect_processes": {"query": {"type": "string"}},
    "mcp_call": {
        "mcp_tool": {"type": "string"},
        "mcp_arguments": {
            "type": "string",
            "description": "JSON-encoded object of arguments for the selected MCP tool, e.g. {}.",
        },
    },
    "request_permission": {
        "path": {"type": "string"},
        "access": {"type": "string", "enum": ["read", "write"]},
    },
    "respond": {"message": {"type": "string"}},
    "finish": {"message": {"type": "string"}},
    "fail": {"message": {"type": "string"}},
}

_NATIVE_TOOL_DESCRIPTIONS = {
    "list_files": "List repository files to understand the workspace structure.",
    "search": "Search repository text and symbols for a query.",
    "read": "Read one repository file using a workspace-relative path.",
    "edit": "Replace one exact text occurrence in an existing file.",
    "apply_patch": "Apply an audited unified diff to one or more existing repository files.",
    "create": "Create a new file with complete content.",
    "move_file": "Move one UTF-8 file within the repository without overwriting.",
    "copy_file": "Copy one UTF-8 file within the repository without overwriting.",
    "delete_path": "Delete one repository file or one empty directory.",
    "git_status": "Read the current Git branch and working-tree status.",
    "git_diff": "Read a repository or file diff, optionally against a revision.",
    "git_log": "Read recent Git commit history.",
    "git_branch": "List branches, or create and switch to one named branch.",
    "git_commit": "Commit only files changed by RAgent in the current turn.",
    "git_restore": "Restore one current-turn RAgent file from a Git revision.",
    "run_tests": "Run the project's test command and collect verification evidence.",
    "run_command": "Run an audited build, check, launch, or system command.",
    "start_terminal": "Start a long-running audited command and return a terminal session ID.",
    "poll_terminal": "Read new output and status from a running terminal session.",
    "write_terminal": "Send input to a running terminal session.",
    "stop_terminal": "Stop a running terminal session.",
    "inspect_processes": (
        "Inspect real visible desktop windows and owning process IDs; query may be empty."
    ),
    "mcp_call": "Call one of the configured read-only MCP project tools.",
    "request_permission": "Request access to an exact path outside the workspace.",
    "respond": "Answer the user when no workspace tool is required.",
    "finish": "Finish only after every task requirement has evidence.",
    "fail": "Stop with a concrete, non-retryable failure explanation.",
}


def _native_tools(
    *, responses_api: bool, allowed_actions: set[str] | None = None
) -> list[dict[str, Any]]:
    """Return one strict function schema per RAgent action."""
    tools: list[dict[str, Any]] = []
    for name, action_fields in _NATIVE_TOOL_FIELDS.items():
        if allowed_actions is not None and name not in allowed_actions:
            continue
        properties = {
            "rationale": {"type": "string"},
            **action_fields,
        }
        if name in {"respond", "finish"}:
            claim_schema = StudioClaim.model_json_schema()
            claim_schema["additionalProperties"] = False
            claim_schema["required"] = list(claim_schema["properties"])
            for field in claim_schema["properties"].values():
                field.pop("default", None)
            properties["claims"] = {"type": "array", "items": claim_schema, "maxItems": 20}
        parameters = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        function = {
            "name": name,
            "description": _NATIVE_TOOL_DESCRIPTIONS[name],
            "parameters": parameters,
            "strict": True,
        }
        tools.append(
            {"type": "function", **function}
            if responses_api
            else {"type": "function", "function": function}
        )
    return tools


def _native_decision(name: str, arguments: str | dict[str, Any]) -> StudioDecision:
    if name == "submit_decision":
        payload = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        return StudioDecision.model_validate(payload)
    if name not in _NATIVE_TOOL_FIELDS:
        raise ValueError(f"Unknown native tool: {name}")
    payload = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    if name == "mcp_call" and isinstance(payload.get("mcp_arguments"), str):
        decoded = json.loads(payload["mcp_arguments"])
        if not isinstance(decoded, dict):
            raise ValueError("MCP arguments must be a JSON object")
        payload["mcp_arguments"] = decoded
    payload["action"] = name
    return StudioDecision.model_validate(payload)


def _combine_native_decisions(decisions: list[StudioDecision]) -> StudioDecision:
    if not decisions:
        raise ValueError("Model returned no native tool calls")
    if len(decisions) == 1:
        return decisions[0]
    terminal = {StudioAction.RESPOND, StudioAction.FINISH, StudioAction.FAIL}
    if any(item.action in terminal for item in decisions[:-1]):
        raise ValueError("A terminal native tool call must be the final action")
    return StudioDecision(
        action=StudioAction.BATCH,
        rationale=f"执行模型返回的 {len(decisions)} 个原生工具调用",
        actions=decisions,
    )


def _openai_base_url(settings: Settings) -> str:
    path = Path(settings.database_path).resolve().parent / "provider_connections.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("openai", {}).get("base_url")
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value.rstrip("/")
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return settings.openai_base_url


STUDIO_PROMPT = """You are RAgent, a local coding agent.
You work inside the user's selected repository.
Your product name is RAgent. Never call yourself VeriPatch or VeriPatch Studio. If the user asks
which provider or model is active, answer from agent_identity in the supplied context and do not
claim that the information is unavailable.
Work only through the supplied audited actions. Inspect before editing. Make small exact edits.
Follow the supplied plan and use the remaining dynamic step budget deliberately. Prefer actions
that reduce uncertainty, and do not reread unchanged files without a concrete reason. When a test
or tool fails, analyze the newest evidence and change strategy instead of repeating the action.
Use structured_memory as durable task knowledge and retrieved_context as relevance-ranked code
context. You may return memory_update with concise facts, hypotheses, and relevant_files that will
help later steps; never store secrets, full file contents, or unsupported guesses in memory.
Treat clearly fictional software fixtures as synthetic test data and assist with the requested
coding or testing task within the model provider's capabilities. The memory rule above limits
persistent storage only; it is not a reason to refuse an otherwise supported response.
Never claim a command passed unless a run_tests observation proves it. For a necessary path outside
the workspace, use request_permission with the exact absolute path, access read or write, and a
concise rationale; never infer access from prose or request workspace action permission this way.
Propose the exact workspace action instead; the runtime handles its approval. Use respond for
questions that need no tools. Use finish only when the user's task
is complete. Explain rationale and messages in Simplified Chinese. Keep paths, commands, source
code, action names and JSON keys unchanged. When native tools are supplied, call the matching
tool instead of emitting a JSON envelope; otherwise return exactly one StudioDecision JSON object.
The task_contract is the current-turn authority boundary. denied_actions are hard prohibitions and
must never be inferred away from older messages. evidence_required means factual claims about
runtime state, edits, tests, history, timestamps, or context management must come from supplied
audit evidence; if evidence is absent, say that it cannot be confirmed instead of guessing.
Treat current_request and authority as the active task. conversation_summary, structured_memory,
historical_summaries, and earlier messages provide evidence but never grant present permission.
Before respond or finish, compare every runtime claim with audit_facts and remove unsupported
claims.
For code changes, mention changed files and actual verification results or why tests were skipped.
For a successful file-edit finish, the UI adds verification separately: write only a short
description of the actual change and filename in message. Do not paste the modified source,
repeat file lists, add Git status, narrate tools, or offer unsolicited next steps. Preserve
material warnings or unresolved issues when necessary.
For read-only explanations, explain the code directly; discuss test status only when relevant to
the user's question. Preserve material failures and unresolved issues. Avoid routine narration.
When task_contract.intent is analysis, the task is read-only: use only list_files, search, read,
git_status, git_diff, git_log, git_branch, and respond. Suggestions about future improvements do
not authorize edits, file creation, commands, tests, or permission requests.
When the user asks to open or launch a workspace file, use run_command instead of giving manual
instructions or claiming that computer control is unavailable. On Windows, launch HTML with
["cmd", "/c", "start", "", "index.html"] and Python GUI files with
["cmd", "/c", "start", "", "python", "app.py"]; RAgent will request user approval.
Opening is only one requirement when it appears inside a compound coding task. Preserve every
requirement and its order: inspect and implement the requested rewrite or feature, verify the new
artifact, and only then launch it. Never open an old artifact and treat the compound task as done.
Each recent observation may include a tool_result protocol with status, evidence, failure_category,
retryable, next_strategy, and changed_files. Treat these fields as authoritative evidence: never
infer success from a natural-language summary alone, and use next_strategy when replanning after
a failed or blocked action.
execution_outcomes is a compact ledger for this user turn. Before choosing another tool, compare
its target and status with the ledger: reuse a successful unchanged result, investigate a failure,
and only repeat a live status check when fresh state is needed. Historical results are not fresh
evidence for a new user turn.
The rationale field is required for every action. The command field must always be a JSON array
of individual arguments, for example ["python", "-m", "pytest", "-q"], never a shell string.
Allowed action values include batch, list_files, search, read, edit, apply_patch, create,
move_file, copy_file, delete_path, git_status, git_diff, git_log, git_branch, git_commit,
git_restore,
run_tests, run_command, start_terminal, poll_terminal, write_terminal, stop_terminal,
inspect_processes, request_permission,
respond, finish, and fail. Use create with path and content only
for a new file. Prefer batch with 2-6 ordered actions when the next operations are already known;
this lets RAgent execute them in one model round trip. Do not nest batch actions.
The create action is an available file-writing tool. When the user asks you to build or add files,
use create/edit in the workspace; never merely paste code and ask the user to save it manually.
Prefer apply_patch over multiple edit calls when a coherent change touches several locations
or files.
Use the structured file and Git actions instead of shell commands when they cover the operation.
Never use delete_path for a non-empty directory. Use git_commit only when the user asked for a
commit; it commits only files changed by RAgent in this turn. Use git_restore only for a file
RAgent changed in this turn and only when the user explicitly requested rollback.
Use start_terminal for long-running servers, watchers, or interactive processes; use the returned
terminal_id with poll_terminal/write_terminal/stop_terminal instead of starting duplicates.
These actions are executed by the RAgent runtime after you return the JSON decision. They are real
workspace tools even though you invoke them by choosing an action in JSON. Never fail by claiming
that file or command tools are unavailable.
Use run_tests for
pytest and run_command for other allowlisted build, lint, type-check and test commands.
Put command details, runtime mechanics, permission
internals, step budgets, and recovery details in tool evidence rather than the user-facing message.
For a successful simple action such as opening a file, one short sentence is enough.
Before finish, check the changed files, verification evidence, and final-review information in the
context. Report the root cause, changed files, verification result, and any remaining risk.
Write the final message for a human, not for the runtime: never paste a long command, script body,
JSON payload, internal budget, or raw tool protocol into the answer. Summarize what changed and
what the evidence proves. If the task is already complete, do not repeat completed work merely
because the user says continue; acknowledge completion and wait for a concrete new objective.
If a test command reports an environment problem, do not repeat the identical command. Continue
with code inspection or explain the one concrete environment requirement to the user. The respond,
finish and fail actions must include a non-empty message for the user.
For edit, always put path, old_text and new_text at the top level. Perform exactly one exact edit
per decision; do not return an edits array or rename path to file."""

COMPAT_CHAT_PROMPT = """You are RAgent, a local coding agent with real workspace tools. Continue
the user's task autonomously and reply in Simplified Chinese. Output exactly one JSON object and
no Markdown. Always include action and rationale. Available actions and required fields:
- list_files: action,rationale
- search: action,rationale,query
- read: action,rationale,path
- create: action,rationale,path,content (content must contain the complete new file)
- edit: action,rationale,path,old_text,new_text (one exact replacement)
- move_file or copy_file: action,rationale,path,destination
- delete_path: action,rationale,path (one file or empty directory only)
- git_status or git_log: action,rationale
- git_diff: action,rationale,path|null,revision|null
- git_branch: action,rationale,branch|null (null lists branches)
- git_commit: action,rationale,commit_message
- git_restore: action,rationale,path,revision
- run_tests or run_command: action,rationale,command (JSON array of argv strings)
- request_permission: action,rationale,path,access (read or write; external paths only)
- respond, finish, or fail: action,rationale,message
Use create/edit instead of pasting code in message or asking the user to save it. Do not finish
while the task contract has unmet completion conditions; if the target or requirements are unclear,
ask one focused clarification.
After changing a file, run a suitable verification action, then finish with a concise result. Do not claim tools are
unavailable and do not reread an unchanged file when existing evidence is present."""


def _response_instructions(context: dict[str, Any]) -> str:
    """Compile the current style once for every provider/protocol entry point."""
    style = context.get("response_style", {})
    mode = style.get("mode", "standard") if isinstance(style, dict) else style
    modes = {
        "concise": "Lead with the answer in 1-3 sentences. For a simple question, give only the direct conclusion and one necessary clarification. Do not restate the question, add background, headings, bullet lists, examples, caveats, workspace status, or follow-up offers unless the user explicitly asks for them. If the user asks for an example or steps, provide only the requested minimum.",
        "standard": "Give the answer, its main reasoning, and relevant practical details.",
        "teaching": "Teach step by step: explain the intuition, walk through the example, and clarify the key pitfall.",
    }
    return (
        "Current response style: " + modes.get(mode, modes["standard"])
        + " Apply this style to this reply, not the style of previous answers. "
        "The user's explicit length and example-count requests take precedence. "
        "For ordinary explanations omit unrelated workspace, command, and test status. "
        "Preserve requested content, material failures, and uncertainty; style changes presentation, not facts or permissions."
        " Write message as the complete, concise user-facing answer. Do not rely on claims"
        " to add any text to the answer or repeat every file as a separate sentence."
        " For explanations of observed file contents or purpose, provide evidence references in claims:"
        " [{kind: observation|fact|inference|unknown, text: ..., observation_id: integer or null}]."
        " Use observation for file listings and read results (including empty files): cite the"
        " observation_id; the runtime audits the reference separately from message."
        " Do not encode operation summaries as fact quotes."
        " A fact must cite its observation_id and match a literal excerpt of file content "
        "or an exact entry in a file listing. Built-in read/files and successful MCP "
        "read_project_file/list_project_files results are equivalent evidence sources. "
        "For MCP observations, evidence is in payload.result. "
        "A listing proves a listed path exists, not the file's contents or purpose."
        " File names, summaries and previous assistant messages do not establish purpose."
        " Use inference for explanations derived from code or names, unknown for missing information."
        " Keep claims short. They are audit references, not separate answer paragraphs."
        " Do not cite tool descriptions or other metadata unless the user asks about them."
        " General knowledge answers and routine operation summaries may use message with empty claims."
    )


def _plain_context(value: Any, indent: int = 0) -> str:
    """Render nested context without a JSON envelope for restrictive compatible proxies."""
    prefix = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.extend((f"{prefix}{key}:", _plain_context(item, indent + 1)))
            else:
                rendered = _plain_context(item, indent + 1).lstrip()
                lines.append(f"{prefix}{key}: {rendered}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.extend((f"{prefix}-", _plain_context(item, indent + 1)))
            else:
                lines.append(f"{prefix}- {_plain_context(item).strip()}")
        return "\n".join(lines) or f"{prefix}(none)"
    if value is None:
        return f"{prefix}(none)"
    text = str(value)
    if "\n" in text:
        return f"{prefix}|\n" + "\n".join(f"{prefix}  {line}" for line in text.splitlines())
    return f"{prefix}{text}"


def parse_studio_decision(content: str) -> StudioDecision:
    """Parse structured output, tolerating a provider's prose around one JSON object."""
    text = (content or "").strip()
    try:
        return StudioDecision.model_validate_json(text)
    except ValueError as original_error:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text, index)
                return StudioDecision.model_validate(payload)
            except (json.JSONDecodeError, ValueError):
                continue
        raise original_error


class StudioProviderModel:
    # Network clients below have connect/read/write timeouts. The Agent must not
    # add a shorter wall-clock timeout: a long create/edit JSON may be streaming
    # useful bytes for over a minute without ever being idle.
    manages_request_timeout = True

    def restore_tool_continuation(
        self, state: dict[str, Any], *, session_id: str, turn_observation_start: int
    ) -> None:
        """Resume a pending provider tool call after a permission round trip."""
        if state.get("protocol") == "responses-tools":
            self._pending_native_call = (
                str(state["response_id"]), str(state["call_id"]),
                int(state["prior_observation_id"]),
            )
        elif state.get("protocol") == "chat-tools":
            self._pending_chat_call = (
                str(state["call_id"]), str(state["name"]),
                str(state["arguments"]), int(state["prior_observation_id"]),
            )
        self._decision_turn = (session_id, turn_observation_start)

    def __init__(
        self,
        provider: str,
        settings: Settings,
        *,
        client: Any | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self._last_protocol = "unknown"
        self._fallback_reason: str | None = None
        self._request_started = 0.0
        self._decision_turn: tuple[object, object] | None = None
        self._pending_native_call: tuple[str, str, int] | None = None
        self._pending_chat_call: tuple[str, str, str, int] | None = None
        self.api_key: str | None = None
        self.base_url = (
            (
                "https://api.openai.com/v1"
                if provider == "openai_official"
                else _openai_base_url(settings).rstrip("/")
            )
            if provider in {"openai", "openai_official"}
            else settings.deepseek_base_url.rstrip("/")
        )
        if client is not None:
            self.client = client
            return
        from openai import AsyncOpenAI

        key = load_api_key(provider)
        if not key:
            raise ValueError(f"{provider} API key is required")
        self.api_key = key
        kwargs: dict[str, Any] = {
            "api_key": key,
            "timeout": 75.0,
            "max_retries": 1,
        }
        if provider == "deepseek":
            kwargs["base_url"] = settings.deepseek_base_url
        else:
            kwargs["base_url"] = self.base_url
        self.client = AsyncOpenAI(**kwargs)

    @property
    def model_name(self) -> str:
        return self.settings.deepseek_model if self.provider == "deepseek" else self.settings.model

    async def classify_intent(
        self, messages: list[dict[str, str]], user_message: str
    ) -> SemanticIntentAssessment:
        """Classify ambiguous language separately from the action-generation loop."""
        system = (
            "You are the primary semantic-understanding layer for a coding agent. Interpret the "
            "whole current utterance in its recent dialogue context before considering keywords. "
            "Return JSON only with intent, confidence, requires_clarification, rationale, "
            "clarification_question, dialogue_act, objectives, questions, requested_actions, "
            "prohibited_actions, "
            "conditions, references, corrections. All list fields must be arrays of strings, "
            "not objects. Use [] for absent lists and an empty string for absent "
            "clarification_question; never null. "
            "Allowed intents: answer, analysis, change, verify, launch_only, install, execute. "
            "Distinguish asking whether an action happened or succeeded from requesting that "
            "action. A question about test status is not permission to run tests. Discussing, "
            "questioning, or evaluating a possible change is answer/analysis, not change. "
            "change requires an actual user instruction to modify files. verify, "
            "launch_only, install, and execute require an actual instruction to perform that "
            "side effect. Preserve every explicit prohibition. Resolve pronouns, corrections, "
            "ellipses, and numbered replies from recent messages. If a reference or effect remains "
            "ambiguous, set requires_clarification=true. requested_actions and prohibited_actions "
            "use exact tool names: read, search, create, edit, apply_patch, move_file (rename), "
            "copy_file, delete_path, run_tests, run_command, git_commit, git_restore, respond. "
            "For every side-effect instruction provide its requested_actions and objectives. "
            "A rename specifies a source and destination; it is a complete change objective. "
            "Starting an interview, roleplay, teaching or other conversational workflow is "
            "answer with requested_actions=[respond], not execute. execute means external "
            "tool side effects. Interpret follow-ups using the selected workflow context. "
            "When clarification is necessary, clarification_question must be one short natural "
            "question in the user's language about the specific missing information. "
            "For language-neutral inputs such as numbers, use the recent dialogue language, "
            "defaulting to Chinese when no language is established. "
            "Permission to edit a file does not specify what edit to make. Never invent a "
            "version bump. If no change objective exists in the dialogue, ask what should change. "
            "A bare number selects an option only when a relevant unanswered choice exists; "
            "otherwise ask what the number refers to. Do not replay a completed task."
        )
        context = json.dumps(
            {"recent_messages": messages, "current_message": user_message},
            ensure_ascii=False,
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.api_key is not None and self.base_url != "https://api.openai.com/v1":
            response = await asyncio.to_thread(self._raw_compatible_chat, payload)
        else:
            response = await self.client.chat.completions.create(**payload)
        content = getattr(response.choices[0].message, "content", "") or ""
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("Intent classifier returned no JSON object")
        return SemanticIntentAssessment.model_validate_json(match.group(0))

    @staticmethod
    def _usage(response: Any) -> tuple[int, int, int, int]:
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
        output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
        cached = getattr(input_details, "cached_tokens", 0)
        reasoning = getattr(output_details, "reasoning_tokens", 0)
        if not reasoning:
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", 0)
        return input_tokens or 0, cached or 0, output_tokens or 0, reasoning or 0

    async def decide(self, context: dict[str, Any]) -> StudioReply:
        decision_turn = (context.get("session_id"), context.get("turn_observation_start"))
        if decision_turn != self._decision_turn:
            self._decision_turn = decision_turn
            self._pending_native_call = None
            self._pending_chat_call = None
        self._last_protocol = "unknown"
        self._fallback_reason = None
        self._request_started = time.perf_counter()
        prompt = json.dumps(context, ensure_ascii=False)
        compatible_context = _plain_context(context)
        response_instructions = _response_instructions(context)
        responses: list[Any] = []
        decision: StudioDecision | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            request_prompt = prompt
            request_compatible = compatible_context
            if attempt:
                request_prompt += (
                    "\n\nYour previous response could not be parsed. Return only one valid "
                    "StudioDecision JSON object. Do not include prose or Markdown outside JSON. "
                    "For implementation work, use create/edit now; do not ask the user "
                    "to save code."
                )
                request_compatible += (
                    "\nCorrection: return one JSON object only. For implementation work, use "
                    "create/edit instead of putting source code in message. If files changed, "
                    "run verification yourself instead of asking the user to run it."
                )
            try:
                response, decision = await self._request_decision(
                    request_prompt, request_compatible, response_instructions
                )
                if decision is None:
                    raise ValueError("Model returned no structured decision")
                allowed = context.get("available_actions")
                if isinstance(allowed, list) and any(
                    action.action.value not in allowed
                    for action in (decision.actions if decision.action is StudioAction.BATCH else [decision])
                ):
                    raise ValueError("Model selected an action unavailable in this task phase")
                decision = self._promote_execution_handoff(decision, context)
                if self._is_code_handoff(decision):
                    raise ValueError(
                        "Implementation must use create/edit instead of asking the user "
                        "to save code"
                    )
                if self._is_false_tool_unavailable_failure(decision):
                    raise ValueError("RAgent workspace tools are available to this decision")
                if self._is_unverified_execution_handoff(decision, context):
                    raise ValueError(
                        "Changed workspace files must be verified with run_command or run_tests"
                    )
                self._pending_native_call = None
                self._pending_chat_call = None
                if self._last_protocol == "responses-tools":
                    calls = [
                        item for item in getattr(response, "output", [])
                        if getattr(item, "type", None) == "function_call"
                    ]
                    response_id = getattr(response, "id", None)
                    call_id = getattr(calls[0], "call_id", None) if len(calls) == 1 else None
                    latest = context.get("latest_tool_result")
                    latest_id = latest.get("observation_id", -1) if isinstance(latest, dict) else -1
                    if response_id and call_id:
                        self._pending_native_call = (str(response_id), str(call_id), latest_id)
                elif self._last_protocol == "chat-tools":
                    calls = getattr(response.choices[0].message, "tool_calls", None) or []
                    call = calls[0] if len(calls) == 1 else None
                    function = getattr(call, "function", None)
                    call_id = getattr(call, "id", None)
                    name = getattr(function, "name", None)
                    arguments = getattr(function, "arguments", None)
                    latest = context.get("latest_tool_result")
                    latest_id = latest.get("observation_id", -1) if isinstance(latest, dict) else -1
                    if call_id and isinstance(name, str) and isinstance(arguments, str):
                        self._pending_chat_call = (str(call_id), name, arguments, latest_id)
                responses.append(response)
                break
            except (ValueError, TypeError) as exc:
                last_error = exc
                decision = None
                self._pending_native_call = None
                self._pending_chat_call = None
            except Exception as exc:
                self._attach_protocol_error(exc)
                raise
        if decision is None:
            error = RuntimeError("模型连续三次未返回有效的结构化动作，请重试本轮。")
            for name in ("raw_response_preview", "response_id", "response_shape"):
                value = getattr(last_error, name, None)
                if value:
                    setattr(error, name, value)
            self._attach_protocol_error(error)
            raise error from last_error
        token_totals = [0, 0, 0, 0]
        for response in responses:
            for index, value in enumerate(self._usage(response)):
                token_totals[index] += value
        return StudioReply(
            decision=decision,
            input_tokens=token_totals[0],
            cached_input_tokens=token_totals[1],
            output_tokens=token_totals[2],
            reasoning_tokens=token_totals[3],
            model=getattr(responses[-1], "model", self.model_name),
            protocol=self._last_protocol,
            fallback_reason=self._fallback_reason,
            response_id=(str(getattr(responses[-1], "id", "")) or None),
            tool_continuation=(
                {
                    "protocol": "responses-tools",
                    "response_id": self._pending_native_call[0],
                    "call_id": self._pending_native_call[1],
                    "prior_observation_id": self._pending_native_call[2],
                }
                if self._pending_native_call else (
                    {
                        "protocol": "chat-tools",
                        "call_id": self._pending_chat_call[0],
                        "name": self._pending_chat_call[1],
                        "arguments": self._pending_chat_call[2],
                        "prior_observation_id": self._pending_chat_call[3],
                    }
                    if self._pending_chat_call else None
                )
            ),
            latency_ms=round((time.perf_counter() - self._request_started) * 1000),
            status_code=200,
        )

    def _attach_protocol_error(self, exc: Exception) -> None:
        exc.__dict__.update(
            protocol=self._last_protocol,
            fallback_reason=self._fallback_reason,
            latency_ms=round((time.perf_counter() - self._request_started) * 1000),
        )

    @staticmethod
    def _is_code_handoff(decision: StudioDecision) -> bool:
        if decision.action is not StudioAction.RESPOND or not decision.message:
            return False
        message = decision.message
        return any(
            marker in message
            for marker in ("请将以下代码保存", "请保存为", "```python", "\nimport tkinter")
        )

    @staticmethod
    def _is_false_tool_unavailable_failure(decision: StudioDecision) -> bool:
        if decision.action not in {StudioAction.FAIL, StudioAction.RESPOND}:
            return False
        text = " ".join(filter(None, (decision.rationale, decision.message)))
        return bool(
            re.search(
                r"(?:没有|未提供|无法使用|不能使用)(?:可用的)?(?:文件操作工具|工作区工具|工具调用能力)|"
                r"(?:工作区工具|文件操作工具)(?:不可用|未提供)|"
                r"(?:没有|未提供)(?:可用的)?(?:文件|命令).{0,16}工具|"
                r"\b(?:tools?|file tools?|workspace tools?)\s+(?:are\s+)?"
                r"(?:unavailable|not available)\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_unverified_execution_handoff(decision: StudioDecision, context: dict[str, Any]) -> bool:
        if decision.action is not StudioAction.RESPOND:
            return False
        completion_state = context.get("completion")
        if (
            isinstance(completion_state, dict)
            and completion_state.get("requires_commands") is False
        ):
            return False
        contract = context.get("task_contract") or {}
        state = context.get("task_state") or {}
        if contract.get("intent") in {"answer", "analysis"}:
            return False
        if state.get("verification_policy") in {"skipped_by_user", "not_applicable"}:
            return False
        denied = set(contract.get("denied_actions", [])) | set(state.get("denied_actions", []))
        if {"run_command", "run_tests"} & denied:
            return False
        changed_files = context.get("changed_files")
        verification = context.get("verification")
        verified = (
            verification.get("verification_passed") if isinstance(verification, dict) else False
        )
        return bool(changed_files) and not bool(verified)

    @classmethod
    def _promote_execution_handoff(
        cls, decision: StudioDecision, context: dict[str, Any]
    ) -> StudioDecision:
        if not cls._is_unverified_execution_handoff(decision, context) or not decision.message:
            return decision
        match = re.search(
            r"(?i)\b(?:python|python\.exe|py)\s+([\w./\\-]+\.py)\b",
            decision.message,
        )
        if not match:
            return decision
        target = match.group(1).replace("\\", "/")
        changed_files = {str(path).replace("\\", "/") for path in context.get("changed_files", [])}
        if target not in changed_files:
            return decision
        return StudioDecision(
            action=StudioAction.RUN_COMMAND,
            rationale="模型建议运行工作区脚本；RAgent 自动转换为经过审计的验证动作。",
            command=["python", target],
        )

    async def _request_decision(
        self, prompt: str, compatible_context: str, response_instructions: str = ""
    ) -> tuple[Any, StudioDecision | None]:
        try:
            request_context = json.JSONDecoder().raw_decode(prompt)[0]
        except ValueError:
            request_context = {}
        if not isinstance(request_context, dict):
            request_context = {}
        allowed = request_context.get("available_actions")
        allowed_actions = set(allowed) if isinstance(allowed, list) else None
        if self.provider in {"openai", "openai_official"}:
            is_official_openai = self.base_url == "https://api.openai.com/v1"
            # Third-party gateways routinely expose a partial /responses route
            # that returns HTTP 200 with an incompatible SSE shape. Their stable
            # interoperability contract is Chat Completions, so do not probe
            # Responses on every new process or Base URL.
            if not is_official_openai or (
                self.provider != "openai_official"
                and _OPENAI_PROTOCOL_CACHE.get(self.base_url) == "chat"
            ):
                response, decision = await self._request_openai_chat(
                    compatible_context, response_instructions, allowed_actions, request_context
                )
            else:
                self._last_protocol = "responses-tools"
                request: dict[str, Any] = dict(
                    model=self.model_name,
                    reasoning={"effort": self.settings.reasoning_effort},
                    instructions=STUDIO_PROMPT + "\n" + response_instructions,
                    input=prompt,
                    tools=_native_tools(responses_api=True, allowed_actions=allowed_actions),
                    tool_choice="required",
                    parallel_tool_calls=True,
                )
                latest = request_context.get("latest_tool_result")
                if self._pending_native_call and isinstance(latest, dict):
                    response_id, call_id, prior_observation_id = self._pending_native_call
                    if latest.get("observation_id", -1) > prior_observation_id:
                        request["previous_response_id"] = response_id
                        request["input"] = [{
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(latest, ensure_ascii=False),
                        }]
                response = await self.client.responses.create(**request)
                decisions = [
                    _native_decision(item.name, item.arguments)
                    for item in getattr(response, "output", [])
                    if getattr(item, "type", None) == "function_call"
                ]
                decision = _combine_native_decisions(decisions)
                _OPENAI_PROTOCOL_CACHE[self.base_url] = "responses-tools"
        elif self.provider == "deepseek" and self.model_name == "deepseek-v4-flash":
            response, decision = await self._request_openai_chat(
                compatible_context, response_instructions, allowed_actions, request_context
            )
        elif self.provider == "deepseek":
            self._last_protocol = "deepseek-chat-json"
            effort = self.settings.reasoning_effort
            if effort not in {"low", "high", "max"}:
                effort = "high"
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": STUDIO_PROMPT + "\n" + response_instructions},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                reasoning_effort=effort,
            )
            decision = self._parse_response_decision(response.choices[0].message.content, response)
        else:
            self._last_protocol = "deepseek-responses-json-schema"
            response = await self.client.responses.create(
                model=self.model_name,
                instructions=STUDIO_PROMPT + "\n" + response_instructions,
                input=prompt,
                reasoning={"effort": self.settings.reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "studio_decision",
                        "schema": StudioDecision.model_json_schema(),
                    }
                },
            )
            decision = self._parse_response_decision(response.output_text, response)
        return response, decision

    async def _request_openai_chat(
        self, compatible_context: str, response_instructions: str = "",
        allowed_actions: set[str] | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> tuple[Any, StudioDecision]:
        use_tools = _OPENAI_PROTOCOL_CACHE.get(self.base_url) != "chat-json"
        self._last_protocol = "chat-tools" if use_tools else "chat-json"
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            # Session reasoning controls must reach compatible gateways too.
            # Previously this was only sent to the official Responses API, so
            # choosing "low" in the UI had no effect for an OpenAI proxy.
            "reasoning_effort": self.settings.reasoning_effort,
            "messages": [
                {
                    "role": "user",
                    "content": COMPAT_CHAT_PROMPT + "\n" + response_instructions + "\nContext:\n" + compatible_context,
                }
            ],
        }
        if use_tools:
            kwargs["tools"] = _native_tools(
                responses_api=False, allowed_actions=allowed_actions
            )
            kwargs["tool_choice"] = "required"
            latest = (request_context or {}).get("latest_tool_result")
            if self._pending_chat_call and isinstance(latest, dict):
                call_id, name, arguments, prior_observation_id = self._pending_chat_call
                if latest.get("observation_id", -1) > prior_observation_id:
                    kwargs["messages"].extend([
                        {
                            "role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": call_id, "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }],
                        },
                        {
                            "role": "tool", "tool_call_id": call_id,
                            "content": json.dumps(latest, ensure_ascii=False),
                        },
                    ])
        else:
            kwargs["response_format"] = {"type": "json_object"}

        async def send(payload: dict[str, Any]) -> Any:
            if self.api_key is not None and self.base_url != "https://api.openai.com/v1":
                return await asyncio.to_thread(self._raw_compatible_chat, payload)
            return await self.client.chat.completions.create(**payload)

        fallback = dict(kwargs)
        fallback.pop("tools", None)
        fallback.pop("tool_choice", None)
        fallback["messages"] = kwargs["messages"][:1]
        fallback["response_format"] = {"type": "json_object"}
        try:
            response = await send(kwargs)
        except httpx.HTTPStatusError as exc:
            if not use_tools or self._status_code(exc) not in {400, 404, 422}:
                raise
            response = await send(fallback)
            _OPENAI_PROTOCOL_CACHE[self.base_url] = "chat-json"
            self._last_protocol = "chat-json"
            self._fallback_reason = f"chat-tools-http-{self._status_code(exc)}"
        except httpx.RemoteProtocolError:
            if not use_tools:
                raise
            # A number of OpenAI-compatible gateways close or return an empty
            # stream when tools/tool_choice are present. No action could have
            # been executed because no valid decision was received, so one
            # JSON fallback request is safe.
            response = await send(fallback)
            _OPENAI_PROTOCOL_CACHE[self.base_url] = "chat-json"
            self._last_protocol = "chat-json"
            self._fallback_reason = "chat-tools-remote-protocol-error"

        message = response.choices[0].message
        try:
            decision = self._chat_message_decision(message, response)
        except (ValueError, TypeError):
            if not use_tools or _OPENAI_PROTOCOL_CACHE.get(self.base_url) == "chat-json":
                raise
            # HTTP 200 with empty/plain content is also a capability failure,
            # not a useful Agent response. Downgrade immediately rather than
            # repeating the incompatible tool request three times.
            response = await send(fallback)
            message = response.choices[0].message
            decision = self._chat_message_decision(message, response)
            _OPENAI_PROTOCOL_CACHE[self.base_url] = "chat-json"
            self._last_protocol = "chat-json"
            self._fallback_reason = "chat-tools-invalid-response"

        if _OPENAI_PROTOCOL_CACHE.get(self.base_url) != "chat-json":
            _OPENAI_PROTOCOL_CACHE[self.base_url] = (
                "chat-tools" if getattr(message, "tool_calls", None) else "chat-json"
            )
        self._last_protocol = _OPENAI_PROTOCOL_CACHE[self.base_url]
        return response, decision

    @classmethod
    def _chat_message_decision(cls, message: Any, response: Any) -> StudioDecision:
        tool_calls = getattr(message, "tool_calls", None) or []
        decisions: list[StudioDecision] = []
        for call in tool_calls:
            function = getattr(call, "function", None)
            name = getattr(function, "name", None)
            if not isinstance(name, str):
                continue
            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str):
                try:
                    decisions.append(_native_decision(name, arguments))
                except (ValueError, TypeError) as exc:
                    cls._add_response_evidence(exc, arguments, response)
                    raise
        if decisions:
            return _combine_native_decisions(decisions)
        content = getattr(message, "content", None)
        return cls._parse_response_decision(content or "", response)

    @staticmethod
    def _parse_response_decision(content: str, response: Any) -> StudioDecision:
        try:
            return parse_studio_decision(content)
        except (ValueError, TypeError) as exc:
            StudioProviderModel._add_response_evidence(exc, content, response)
            raise

    @staticmethod
    def _add_response_evidence(exc: Exception, content: str, response: Any) -> None:
        preview = re.sub(r"sk-[A-Za-z0-9_-]{6,}", "[密钥已隐藏]", (content or "").strip())
        exc.raw_response_preview = preview[:1200] or "（空响应）"  # type: ignore[attr-defined]
        response_id = getattr(response, "id", None)
        if response_id:
            exc.response_id = str(response_id)[:160]  # type: ignore[attr-defined]
        stripped = (content or "").lstrip()
        shape = "JSON 文本" if stripped.startswith(("{", "[")) else "普通文本"
        exc.response_shape = shape  # type: ignore[attr-defined]

    def _raw_compatible_chat(self, payload: dict[str, Any]) -> SimpleNamespace:
        request_payload = dict(payload)
        request_payload["stream"] = True
        official_deepseek = self.provider == "deepseek" and self.base_url == "https://api.deepseek.com"
        if official_deepseek:
            request_payload["stream_options"] = {"include_usage": True}
        # Retries belong to the checkpointed Agent layer. Retrying here used to
        # keep an opaque request alive for several minutes and could outlive the
        # Agent's outer timeout without preserving progress.
        with httpx.stream(
            "POST",
            self.base_url + "/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": f"RAgent/{__version__}",
                "X-Title": "RAgent",
            },
            json=request_payload,
            timeout=httpx.Timeout(55, connect=15, write=30, pool=15),
        ) as response:
            response.raise_for_status()
            return self._decode_compatible_stream(response, wait_for_usage=official_deepseek)

    def _decode_compatible_stream(self, response: httpx.Response, *, wait_for_usage: bool = False) -> SimpleNamespace:
        content_type = response.headers.get("content-type", "").casefold()
        if "text/event-stream" not in content_type:
            response.read()
            return self._compatible_response(response.json())

        content: list[str] = []
        model = self.model_name
        usage: dict[str, Any] = {}
        tool_arguments: dict[int, list[str]] = {}
        tool_names: dict[int, str] = {}
        decision_ready = False
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            event = json.loads(raw)
            model = event.get("model") or model
            if event.get("usage"):
                usage = event["usage"]
                if decision_ready:
                    break
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                for tool_call in delta.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    index = int(tool_call.get("index", 0))
                    function = tool_call.get("function") or {}
                    name = function.get("name")
                    arguments = function.get("arguments")
                    if isinstance(name, str):
                        tool_names[index] = name
                    if isinstance(arguments, str):
                        tool_arguments.setdefault(index, []).append(arguments)
                        candidate = "".join(tool_arguments[index])
                        if tool_names.get(index) == "submit_decision":
                            try:
                                parse_studio_decision(candidate)
                            except ValueError:
                                pass
                            else:
                                if wait_for_usage:
                                    decision_ready = True
                                    continue
                                return self._compatible_response(
                                    {
                                        "model": model,
                                        "choices": [
                                            {
                                                "message": {
                                                    "content": "",
                                                    "tool_calls": [
                                                        {
                                                            "function": {
                                                                "name": "submit_decision",
                                                                "arguments": candidate,
                                                            }
                                                        }
                                                    ],
                                                }
                                            }
                                        ],
                                        "usage": usage,
                                    }
                                )
                piece = delta.get("content")
                if isinstance(piece, str):
                    content.append(piece)
                    # Some OpenAI-compatible gateways send a complete JSON
                    # decision but keep the SSE connection open indefinitely.
                    # As soon as the accumulated output validates as an Agent
                    # action, close the response instead of waiting for [DONE].
                    candidate = "".join(content)
                    try:
                        parse_studio_decision(candidate)
                    except ValueError:
                        pass
                    else:
                        if wait_for_usage:
                            decision_ready = True
                            continue
                        return self._compatible_response(
                            {
                                "model": model,
                                "choices": [{"message": {"content": candidate}}],
                                "usage": usage,
                            }
                        )
        if not content and not tool_arguments:
            raise httpx.RemoteProtocolError("Streaming response contained no model output")
        if tool_arguments:
            calls = [
                {
                    "function": {
                        "name": tool_names.get(index, ""),
                        "arguments": "".join(parts),
                    }
                }
                for index, parts in sorted(tool_arguments.items())
            ]
            return self._compatible_response(
                {
                    "model": model,
                    "choices": [{"message": {"content": "", "tool_calls": calls}}],
                    "usage": usage,
                }
            )
        return self._compatible_response(
            {
                "model": model,
                "choices": [{"message": {"content": "".join(content)}}],
                "usage": usage,
            }
        )

    def _compatible_response(self, data: dict[str, Any]) -> SimpleNamespace:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            output_text = data.get("output_text")
            if not isinstance(output_text, str):
                output_text = self._responses_output_text(data.get("output"))
            if output_text:
                choices = [{"message": {"content": output_text}}]
            else:
                error = data.get("error")
                detail = error.get("message") if isinstance(error, dict) else error
                raise httpx.RemoteProtocolError(
                    f"Compatible endpoint returned no choices{f': {detail}' if detail else ''}"
                )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise httpx.RemoteProtocolError("Compatible endpoint returned an invalid choice")
        message = choice.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(content, str):
            content = choice.get("text")
        if (not isinstance(content, str) or not content.strip()) and not tool_calls:
            raise httpx.RemoteProtocolError("Compatible endpoint returned empty model output")
        parsed_calls: list[SimpleNamespace] = []
        for call in tool_calls or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            parsed_calls.append(
                SimpleNamespace(
                    function=SimpleNamespace(
                        name=function.get("name"), arguments=function.get("arguments")
                    )
                )
            )
        usage = data.get("usage") or {}
        return SimpleNamespace(
            model=data.get("model", self.model_name),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content or "", tool_calls=parsed_calls)
                )
            ],
            usage=SimpleNamespace(**usage),
        )

    @staticmethod
    def _responses_output_text(output: Any) -> str | None:
        if not isinstance(output, list):
            return None
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    parts.append(content["text"])
        return "".join(parts) or None

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(exc, "response", None)
        nested = getattr(response, "status_code", None)
        return nested if isinstance(nested, int) else None
