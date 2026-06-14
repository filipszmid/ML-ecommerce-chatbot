"""Google ADK agent definition for the ecommerce chat demo."""

# ADK discovers `root_agent`, and SafeLiteLlm intentionally mirrors the ADK
# LiteLlm class naming while only customizing serialization for local tests.

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents.llm_agent import Agent

from interface.chat.ecommerce_chat.tools import predict_product_category
from master_config import (
    ADK_MODEL,
    AZURE_OPENAI_DEPLOYMENT,
    BEDROCK_MODEL_ID,
    LLM_PROVIDER,
    OLLAMA_API_BASE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    VERTEX_MODEL_ID,
)


def load_instruction() -> str:
    """Load the agent instruction text.

    Returns:
        Instruction text.
    """
    return (Path(__file__).with_name("instructions.txt")).read_text(encoding="utf-8")


try:
    from google.adk.models.lite_llm import LiteLlm
    from pydantic import field_serializer

    class SafeLiteLlm(LiteLlm):
        """LiteLLM wrapper with safe serializer for ADK inspection."""

        @field_serializer("llm_client", check_fields=False)
        def serialize_llm_client(self, client, _info) -> str:
            """Serialize the internal LiteLLM client without leaking internals."""
            _ = client
            return "LiteLLMClient"

except Exception:
    SafeLiteLlm = None  # type: ignore


def build_model() -> object:
    """Build the ADK model configuration.

    Returns:
        ADK model name or LiteLLM wrapper.
    """
    provider = LLM_PROVIDER.lower()
    if provider == "openai":
        model_name = OPENAI_MODEL
        return (
            SafeLiteLlm(model=f"openai/{model_name}")
            if SafeLiteLlm
            else f"openai/{model_name}"
        )
    if provider == "ollama":
        os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_API_BASE or OLLAMA_BASE_URL)
        model_name = OLLAMA_MODEL
        return (
            SafeLiteLlm(model=f"ollama_chat/{model_name}")
            if SafeLiteLlm
            else f"ollama_chat/{model_name}"
        )
    if provider in {"vertex", "gcp", "gemini"}:
        return VERTEX_MODEL_ID
    if provider in {"azure", "azure_openai"}:
        deployment = AZURE_OPENAI_DEPLOYMENT
        return (
            SafeLiteLlm(model=f"azure/{deployment}")
            if SafeLiteLlm
            else f"azure/{deployment}"
        )
    if provider == "bedrock":
        model_id = BEDROCK_MODEL_ID
        return (
            SafeLiteLlm(model=f"bedrock/{model_id}")
            if SafeLiteLlm
            else f"bedrock/{model_id}"
        )
    return ADK_MODEL


root_agent = Agent(
    model=build_model(),
    name="ecommerce_category_agent",
    description=(
        "Conversational agent that collects customer purchase features and "
        "calls the prediction API."
    ),
    instruction=load_instruction(),
    tools=[predict_product_category],
)
