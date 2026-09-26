"""DeepSeek V4 adapter with provider-specific structured output handling."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from veripatch.config import Settings
from veripatch.credentials import load_api_key
from veripatch.domain import AgentDecision
from veripatch.models.base import ModelContext, ModelReply
from veripatch.models.openai import SYSTEM_PROMPT


class DeepSeekModel:
    def __init__(
        self,
        settings: Settings,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        if client is not None:
            self.client = client
            return
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use DeepSeek") from exc
        resolved_key = api_key or load_api_key("deepseek")
        if not resolved_key:
            raise ValueError("DEEPSEEK_API_KEY or a keyring credential is required")
        self.client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=self.settings.deepseek_base_url,
        )

    def _prompt(self, context: ModelContext) -> str:
        schema = AgentDecision.model_json_schema()
        return (
            "Current repair state:\n"
            + context.model_dump_json()
            + "\nReturn one JSON object matching this JSON Schema exactly:\n"
            + json.dumps(schema, ensure_ascii=False)
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        direct = getattr(response, "output_text", None)
        if direct:
            return str(direct)
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    return str(getattr(content, "text", ""))
        return ""

    async def _responses(self, prompt: str) -> tuple[Any, str]:
        response = await self.client.responses.create(
            model=self.settings.deepseek_model,
            instructions=SYSTEM_PROMPT + " Always return valid JSON.",
            input=prompt,
            reasoning={"effort": self.settings.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "agent_decision",
                    "schema": AgentDecision.model_json_schema(),
                }
            },
        )
        return response, self._response_text(response)

    async def _chat_completions(self, prompt: str) -> tuple[Any, str]:
        effort = self.settings.reasoning_effort
        if effort not in {"low", "high", "max"}:
            effort = "high"
        response = await self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + " Return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            reasoning_effort=effort,
        )
        return response, response.choices[0].message.content or ""

    @staticmethod
    def _parse_decision(content: str) -> AgentDecision:
        """Accept one valid decision while tolerating common provider JSON glitches."""
        cleaned = content.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()

        decoder = json.JSONDecoder()
        candidates: list[Any] = []
        try:
            candidates.append(json.loads(cleaned))
        except json.JSONDecodeError:
            cursor = 0
            while True:
                start = cleaned.find("{", cursor)
                if start < 0:
                    break
                try:
                    candidate, end = decoder.raw_decode(cleaned, start)
                except json.JSONDecodeError:
                    cursor = start + 1
                    continue
                candidates.append(candidate)
                cursor = end

        last_error: Exception | None = None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            normalized = dict(candidate)
            if normalized.get("edits") is None:
                normalized["edits"] = []
            try:
                return AgentDecision.model_validate(normalized)
            except ValidationError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ValueError("DeepSeek returned no valid JSON object")

    @staticmethod
    def _usage(response: Any) -> tuple[int, int, int, int]:
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", 0)
        cached_tokens = getattr(input_details, "cached_tokens", None)
        if cached_tokens is None:
            cached_tokens = getattr(usage, "prompt_cache_hit_tokens", 0)
        reasoning_tokens = getattr(output_details, "reasoning_tokens", None)
        if reasoning_tokens is None:
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0)
        return (
            input_tokens or 0,
            cached_tokens or 0,
            output_tokens or 0,
            reasoning_tokens or 0,
        )

    async def decide(self, context: ModelContext) -> ModelReply:
        base_prompt = self._prompt(context)
        prompt = base_prompt
        last_error: Exception | None = None
        response: Any | None = None
        responses: list[Any] = []
        for _ in range(2):
            try:
                if self.settings.deepseek_model == "deepseek-v4-pro":
                    response, content = await self._chat_completions(prompt)
                else:
                    response, content = await self._responses(prompt)
                responses.append(response)
            except Exception as exc:
                raise RuntimeError(f"DeepSeek API failed: {exc}") from exc
            try:
                if not content.strip():
                    raise ValueError("DeepSeek returned empty structured output")
                decision = self._parse_decision(content)
                break
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                prompt = (
                    base_prompt
                    + "\nYour previous response was invalid. Return exactly one corrected JSON "
                    "object with no markdown or trailing text. Validation error: "
                    + str(exc)[:1_000]
                    + "\nPrevious response:\n"
                    + content[:4_000]
                )
        else:
            raise RuntimeError(f"DeepSeek returned invalid AgentDecision: {last_error}")

        usage_totals = [self._usage(item) for item in responses]
        return ModelReply(
            decision=decision,
            input_tokens=sum(item[0] for item in usage_totals),
            cached_input_tokens=sum(item[1] for item in usage_totals),
            output_tokens=sum(item[2] for item in usage_totals),
            reasoning_tokens=sum(item[3] for item in usage_totals),
            model=getattr(response, "model", self.settings.deepseek_model),
            request_id=getattr(response, "id", None),
        )
