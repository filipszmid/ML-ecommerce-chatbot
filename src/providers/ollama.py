"""Ollama provider adapter."""

# The fine-tune signature implements the shared provider interface. QLoRA
# imports stay lazy so normal chat mode does not import training dependencies.

from __future__ import annotations

import json
import urllib.request

from master_config import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.providers.base import LLMMessage, LLMProvider

OLLAMA_TO_HF_BASE_MODEL = {
    "llama3.1": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama3.1:8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama3.2": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "llama3.2:3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "llama3.2:1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tinyllama:1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


class OllamaProvider(LLMProvider):
    """LLM provider backed by local Ollama."""

    provider_name = "ollama"

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout_seconds: int = 30,
    ) -> None:
        """Initialize provider.

        Args:
            base_url: Ollama base URL.
            model: Ollama model name.
            timeout_seconds: Request timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response via Ollama.

        Args:
            messages: Chat messages.
            system_prompt: Optional system prompt.

        Returns:
            Response text.
        """
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(
            {"role": message.role, "content": message.content} for message in messages
        )
        payload = {
            "model": self.model,
            "messages": request_messages,
            "stream": False,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        return raw.get("message", {}).get("content", "")

    def finetune(
        self,
        dataset_path: str,
        epochs: int = 3,
        run_name: str | None = None,
        base_model_id: str | None = None,
        batch_size: int = 4,
        max_seq_length: int = 512,
        max_train_samples: int | None = None,
        clearml_enabled: bool = False,
    ) -> str:
        """Fine-tune the local model via QLoRA.

        Args:
            dataset_path: Path to training data.
            epochs: Number of training epochs.
            run_name: Optional custom run name for tracking.
            base_model_id: Hugging Face/local checkpoint to train.
            batch_size: Per-device training batch size.
            max_seq_length: Maximum sequence length.
            max_train_samples: Optional cap for local smoke/demo runs.
            clearml_enabled: Whether to log the fine-tuning run into ClearML.

        Returns:
            Path to the saved LoRA adapter.
        """
        from src.finetuning.local_qlora import LLMFineTuner

        training_base_model = base_model_id or resolve_hf_base_model(self.model)
        if training_base_model != self.model:
            print(
                "Resolved Ollama model "
                f"{self.model} to Hugging Face base model {training_base_model}"
            )

        tuner = LLMFineTuner(
            base_model_id=training_base_model,
            ollama_model=self.model,
            dataset_path=dataset_path,
            run_name=run_name,
            epochs=epochs,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            max_train_samples=max_train_samples,
            clearml_enabled=clearml_enabled,
        )
        return tuner.train()


def resolve_hf_base_model(ollama_model: str) -> str:
    """Resolve an Ollama model tag to a Hugging Face/local training checkpoint.

    Args:
        ollama_model: Ollama model tag or a Hugging Face/local model id.

    Returns:
        Hugging Face model id or local path usable by Transformers.

    Raises:
        ValueError: If the Ollama tag has no known training checkpoint mapping.
    """
    normalized = ollama_model.strip().lower()
    if normalized in OLLAMA_TO_HF_BASE_MODEL:
        return OLLAMA_TO_HF_BASE_MODEL[normalized]
    if "/" in ollama_model or ollama_model.startswith("."):
        return ollama_model
    raise ValueError(
        f"{ollama_model!r} is an Ollama tag, not a Transformers checkpoint. "
        "Pass --base-model-id with the Hugging Face or local checkpoint used "
        "to train the LoRA adapter."
    )
