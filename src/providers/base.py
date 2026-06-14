"""Base interface for chat LLM providers."""

# Provider implementations share one explicit fine-tuning interface so callers
# can switch backends without adapter-specific argument objects.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMMessage:
    """Chat message passed to LLM providers.

    Args:
        role: Message role.
        content: Message content.
    """

    role: str
    content: str


class LLMProvider:
    """Base interface for LLM providers."""

    provider_name = "base"

    def generate(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response.

        Args:
            messages: Conversation messages.
            system_prompt: Optional system prompt.

        Returns:
            Generated response text.
        """
        raise NotImplementedError

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
        """Fine-tune the provider's underlying model.

        Args:
            dataset_path: Path to training data.
            epochs: Number of training epochs.
            run_name: Optional custom run name for tracking.
            base_model_id: Optional provider-specific base model override.
            batch_size: Per-device training batch size.
            max_seq_length: Maximum sequence length.
            max_train_samples: Optional cap for local smoke/demo runs.
            clearml_enabled: Whether to log the fine-tuning run into ClearML.

        Returns:
            String identifying the fine-tuned artifact or job.
        """
        _ = dataset_path
        _ = epochs
        _ = run_name
        _ = base_model_id
        _ = batch_size
        _ = max_seq_length
        _ = max_train_samples
        _ = clearml_enabled
        raise NotImplementedError("Fine-tuning is not implemented for this provider.")


class DeterministicProvider(LLMProvider):
    """Provider used when no remote LLM is available."""

    provider_name = "deterministic"

    def generate(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
    ) -> str:
        """Return a deterministic response.

        Args:
            messages: Conversation messages.
            system_prompt: Optional system prompt.

        Returns:
            Response text.
        """
        _ = messages
        _ = system_prompt
        return ""
