"""Model adapters for VeriPatch."""

from veripatch.models.base import AgentModel, ModelContext, ModelReply
from veripatch.models.deepseek import DeepSeekModel
from veripatch.models.scripted import DiscountBugDemoModel

__all__ = [
    "AgentModel",
    "DeepSeekModel",
    "DiscountBugDemoModel",
    "ModelContext",
    "ModelReply",
]
