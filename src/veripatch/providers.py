"""Provider registry shared by CLI and API surfaces."""

from __future__ import annotations

from dataclasses import replace

from veripatch.config import Settings
from veripatch.models.base import AgentModel
from veripatch.models.deepseek import DeepSeekModel
from veripatch.models.openai import OpenAIResponsesModel
from veripatch.models.scripted import DiscountBugDemoModel


def create_model(provider: str, settings: Settings) -> AgentModel:
    if provider == "openai":
        return OpenAIResponsesModel(settings)
    if provider == "openai_official":
        return OpenAIResponsesModel(
            replace(settings, openai_base_url="https://api.openai.com/v1"),
            credential_provider="openai_official",
        )
    if provider == "deepseek":
        if settings.deepseek_model in {"deepseek-chat", "deepseek-reasoner"}:
            raise ValueError(
                "Retired DeepSeek model name; use deepseek-v4-flash or deepseek-v4-pro"
            )
        return DeepSeekModel(settings)
    if provider == "scripted-demo":
        return DiscountBugDemoModel()
    raise ValueError(f"Unknown provider: {provider}")
