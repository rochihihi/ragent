"""OpenAI Responses API adapter with typed structured output."""

from __future__ import annotations

import json
from typing import Any

from veripatch.config import Settings
from veripatch.credentials import load_api_key
from veripatch.domain import AgentDecision
from veripatch.models.base import ModelContext, ModelReply

SYSTEM_PROMPT = """You repair Python repositories through a small audited action set.
Choose exactly one next action. Gather evidence before editing. Use exact source text in edits.
Never edit tests or repository metadata. The runtime, not you, decides whether tests pass.
Prefer concise searches and bounded reads. If evidence is insufficient, investigate further.
Finish only after a successful deterministic test observation. Otherwise continue investigating
or fail with a clear reason.
Write rationale, final_summary, and all user-facing explanations in Simplified Chinese.
Keep action names, file paths, search queries, source code, and JSON field names unchanged."""


class OpenAIResponsesModel:
    def __init__(
        self,
        settings: Settings,
        *,
        api_key: str | None = None,
        credential_provider: str = "openai",
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        if client is not None:
            self.client = client
            return
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use this provider") from exc
        resolved_key = api_key or load_api_key(credential_provider)
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
        self.client = AsyncOpenAI(api_key=resolved_key, base_url=self.settings.openai_base_url)

    async def decide(self, context: ModelContext) -> ModelReply:
        payload = context.model_dump(mode="json")
        try:
            response = await self.client.responses.parse(
                model=self.settings.model,
                reasoning={"effort": self.settings.reasoning_effort},
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Current repair state:\n"
                        + json.dumps(payload, ensure_ascii=False),
                    },
                ],
                text_format=AgentDecision,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI Responses API failed: {exc}") from exc
        decision = response.output_parsed
        if decision is None:
            raise RuntimeError("Model returned no structured AgentDecision")
        usage = response.usage
        input_details = getattr(usage, "input_tokens_details", None) if usage else None
        output_details = getattr(usage, "output_tokens_details", None) if usage else None
        return ModelReply(
            decision=decision,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            cached_input_tokens=getattr(input_details, "cached_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            reasoning_tokens=getattr(output_details, "reasoning_tokens", 0),
            model=getattr(response, "model", self.settings.model),
            request_id=getattr(response, "id", None),
        )
