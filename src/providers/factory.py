"""Provider factory."""

# Provider imports stay lazy so optional cloud SDKs are only imported when that
# provider is selected.

from __future__ import annotations

from loguru import logger

from master_config import LLM_PROVIDER
from src.providers.base import DeterministicProvider, LLMProvider


def build_llm_provider(
    provider_name: str | None = None,
    allow_fallback: bool = True,
) -> LLMProvider:
    """Build an LLM provider from configuration.

    Args:
        provider_name: Provider key.
        allow_fallback: Fall back to deterministic provider on setup errors.

    Returns:
        Provider instance. Falls back to deterministic provider if unavailable.
    """
    selected = (provider_name or LLM_PROVIDER).lower()
    try:
        if selected == "ollama":
            from src.providers.ollama import OllamaProvider

            return OllamaProvider()
        if selected == "bedrock":
            from src.providers.bedrock import BedrockProvider

            return BedrockProvider()
        if selected in {"azure", "azure_openai"}:
            from src.providers.azure_openai import AzureOpenAIProvider

            return AzureOpenAIProvider()
        if selected in {"gcp", "vertex", "vertex_ai"}:
            from src.providers.vertex import VertexAIProvider

            return VertexAIProvider()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if not allow_fallback:
            raise
        logger.warning(f"Falling back to deterministic chat provider: {exc}")
    return DeterministicProvider()
