"""Generative AI Fine-Tuning Pipeline using QLoRA."""

# This module orchestrates a training job with many explicit hyperparameters.
# The public constructor and training method are intentionally stable for CLI
# and ClearML pipeline callers.

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from evaluation.tracking.clearml_tracker import ClearMLTracker
from master_config import (
    CLEARML_ENABLED,
    CLEARML_SERVING_BASE_URL,
    CLEARML_SERVING_ENDPOINT,
)
from src.models.common.artifacts import save_json


def register_existing_lora_adapter(
    adapter_path: Path | str,
    ollama_model: str | None,
    base_model_id: str | None,
    run_name: str | None = None,
    clearml_enabled: bool = True,
) -> dict[str, Any]:
    """Register an existing LoRA adapter in ClearML without retraining.

    Args:
        adapter_path: Existing LoRA adapter directory.
        ollama_model: Ollama model tag this adapter is served with.
        base_model_id: Hugging Face or local checkpoint used for training.
        run_name: Optional registration task name.
        clearml_enabled: Whether to register in ClearML.

    Returns:
        Registration manifest.
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_dir}")

    resolved_run_name = run_name or f"register_{adapter_dir.parent.name}"
    tracker = ClearMLTracker(enabled=clearml_enabled)
    tracker.start(
        task_name=resolved_run_name,
        params={
            "adapter_path": str(adapter_dir),
            "base_model_id": base_model_id,
            "ollama_model": ollama_model,
        },
        task_type="training",
        tags=["llm", "finetuning", "qlora", "ollama", "registration"],
    )

    split_dir = _find_existing_split_dir(adapter_dir)
    clearml_dataset_id = None
    if split_dir is not None:
        clearml_dataset_id = tracker.log_dataset(
            dataset_path=split_dir,
            dataset_name="llm_ecommerce_chatbot_sft",
            dataset_version=resolved_run_name,
            tags=["llm", "ecommerce", "sft", "qlora"],
        )
        tracker.upload_artifacts(
            {
                "llm_train_split": split_dir / "train.csv",
                "llm_eval_split": split_dir / "eval.csv",
                "llm_test_split": split_dir / "test.csv",
            }
        )

    finetuned_ollama_model = (
        f"finetuned-{ollama_model.replace(':', '_')}" if ollama_model else None
    )
    serving_endpoint = (
        f"{CLEARML_SERVING_BASE_URL.rstrip('/')}/"
        f"{CLEARML_SERVING_ENDPOINT.strip('/')}-llm"
    )
    adapter_archive = Path(f"{adapter_dir}.zip")
    manifest = {
        "adapter_dir": str(adapter_dir),
        "adapter_archive": str(adapter_archive),
        "base_model_id": base_model_id,
        "clearml_dataset_id": clearml_dataset_id,
        "clearml_llm_serving_endpoint": serving_endpoint,
        "finetuned_ollama_model": finetuned_ollama_model,
        "ollama_model": ollama_model,
        "run_name": resolved_run_name,
    }
    save_json(adapter_dir / "fine_tune_manifest.json", manifest)
    save_json(adapter_dir.parent / "fine_tune_manifest.json", manifest)
    adapter_archive = Path(shutil.make_archive(str(adapter_dir), "zip", adapter_dir))
    save_json(Path("data/runs/llm_finetuning/latest_finetune.json"), manifest)

    tracker.upload_artifacts(
        {
            "lora_adapter_archive": adapter_archive,
            "fine_tune_manifest": adapter_dir.parent / "fine_tune_manifest.json",
        }
    )
    clearml_model_id = tracker.register_output_model(
        model_path=adapter_archive,
        model_name=f"llm-lora-{resolved_run_name}",
        metadata=manifest,
        framework="pytorch",
    )
    if clearml_model_id:
        manifest["clearml_model_id"] = clearml_model_id
        save_json(adapter_dir / "fine_tune_manifest.json", manifest)
        save_json(adapter_dir.parent / "fine_tune_manifest.json", manifest)
        save_json(Path("data/runs/llm_finetuning/latest_finetune.json"), manifest)
        tracker.upload_artifact(
            "fine_tune_manifest_with_model_id",
            adapter_dir.parent / "fine_tune_manifest.json",
        )
    tracker.close()
    return manifest


def _find_existing_split_dir(adapter_dir: Path) -> Path | None:
    """Find existing LLM split files from current or legacy runs.

    Args:
        adapter_dir: Adapter directory.

    Returns:
        Directory containing split CSV files, if present.
    """
    candidates = [
        adapter_dir.parent / "dataset",
        adapter_dir.parent.parent / "dataset",
        Path("data/evals/llm_splits"),
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists():
            return candidate
    return None


class ClearMLTrainerMetricCallback(TrainerCallback):
    """Report Hugging Face Trainer log events into ClearML scalar curves."""

    def __init__(self, tracker: ClearMLTracker, run_name: str) -> None:
        """Initialize callback.

        Args:
            tracker: Active ClearML tracker.
            run_name: Fine-tuning run name.
        """
        self.tracker = tracker
        self.run_name = run_name

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Report Trainer logs at their native global step.

        Args:
            args: Trainer arguments.
            state: Trainer state.
            control: Trainer control object.
            logs: Metric log payload.
            kwargs: Additional Trainer callback payload.
        """
        if not logs:
            return
        iteration = int(getattr(state, "global_step", 0) or 0)
        for metric_name, value in logs.items():
            scalar_value = LLMFineTuner.as_float(value)
            if scalar_value is None:
                continue
            self.tracker.report_scalar_points(
                f"llm_finetuning/live/{metric_name}",
                self.run_name,
                [(iteration, scalar_value)],
            )


class LLMFineTuner:
    """QLoRA Fine-Tuner for Generative AI Chatbot Models."""

    def __init__(
        self,
        base_model_id: str,
        ollama_model: str | None = None,
        dataset_path: str = (
            "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"
        ),
        output_dir: Path | str = "data/runs/llm_finetuning",
        run_name: str | None = None,
        epochs: int = 3,
        batch_size: int = 4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        max_seq_length: int = 512,
        max_train_samples: int | None = None,
        clearml_enabled: bool = CLEARML_ENABLED,
    ) -> None:
        """Initialize the fine-tuning pipeline.

        Args:
            base_model_id: HuggingFace model ID
                (e.g. meta-llama/Meta-Llama-3-8B-Instruct).
            ollama_model: Ollama model tag this adapter is intended to be served with.
            dataset_path: Path or HF dataset identifier.
            output_dir: Directory to save the trained LoRA adapter.
            run_name: Optional custom run name for tracking.
            epochs: Number of training epochs.
            batch_size: Per-device train batch size.
            lora_r: LoRA attention dimension.
            lora_alpha: LoRA alpha parameter.
            max_seq_length: Maximum sequence length used by SFT.
            max_train_samples: Optional cap for local smoke/demo runs.
            clearml_enabled: Whether to log the run into ClearML.
        """
        self.base_model_id = base_model_id
        self.ollama_model = ollama_model
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.run_name = run_name or f"finetune_{base_model_id.split('/')[-1]}"
        self.epochs = epochs
        self.batch_size = batch_size
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.max_seq_length = max_seq_length
        self.max_train_samples = max_train_samples
        self.clearml_enabled = clearml_enabled
        self.split_dir = self.output_dir / self.run_name / "dataset"
        self.clearml_dataset_id: str | None = None
        self.clearml_model_id: str | None = None

        self.tracker = ClearMLTracker(enabled=self.clearml_enabled)
        self.tracker.start(
            task_name=self.run_name,
            params={
                "base_model": self.base_model_id,
                "ollama_model": self.ollama_model,
                "dataset": self.dataset_path,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "max_seq_length": self.max_seq_length,
                "max_train_samples": self.max_train_samples,
            },
            task_type="training",
            tags=["llm", "finetuning", "qlora", "ollama"],
        )
        self.task = self.tracker.task

    def _format_dataset(self, tokenizer: AutoTokenizer) -> dict[str, Dataset]:
        """Load and format the e-commerce dataset for chat SFT.

        Args:
            tokenizer: Tokenizer used to apply the native chat template.

        Returns:
            Train/eval dataset mapping with a `text` column.
        """
        dataset_file = Path(self.dataset_path)
        if dataset_file.exists():
            if dataset_file.suffix.lower() == ".jsonl":
                dataset = load_dataset("json", data_files=str(dataset_file))
            else:
                dataset = load_dataset("csv", data_files=str(dataset_file))
        elif self.dataset_path.startswith("hf://") or "/" in self.dataset_path:
            path = self.dataset_path.replace("hf://datasets/", "")
            dataset = load_dataset(path)
        else:
            dataset = load_dataset("csv", data_files=self.dataset_path)

        train_data = dataset["train"] if "train" in dataset else dataset

        if isinstance(train_data, Dataset):
            splits = train_data.train_test_split(test_size=0.1, seed=42)
            train_dataset = splits["train"]
            eval_dataset = splits["test"]
        else:
            raise ValueError("Unexpected dataset format")

        if self.max_train_samples:
            train_limit = min(self.max_train_samples, len(train_dataset))
            eval_limit = min(max(1, self.max_train_samples // 10), len(eval_dataset))
            train_dataset = train_dataset.select(range(train_limit))
            eval_dataset = eval_dataset.select(range(eval_limit))

        def format_chat_template(example: dict[str, Any]) -> dict[str, str]:
            """Format into conversational instruction-response pairs."""
            if "text" in example and example["text"]:
                return {"text": str(example["text"])}

            instruction = self._first_text_value(
                example,
                ("instruction", "prompt", "question", "user", "input"),
            )
            response = self._first_text_value(
                example,
                ("response", "answer", "assistant", "output", "completion"),
            )
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            if getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                text = f"<|user|>\n{instruction}\n<|assistant|>\n{response}"
            return {"text": text}

        train_dataset = train_dataset.map(format_chat_template)
        eval_dataset = eval_dataset.map(format_chat_template)

        self.split_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.split_dir / "train.csv"
        eval_path = self.split_dir / "eval.csv"
        train_dataset.to_csv(train_path)
        eval_dataset.to_csv(eval_path)

        self.tracker.upload_artifacts(
            {
                "llm_train_split": train_path,
                "llm_eval_split": eval_path,
            }
        )
        self.clearml_dataset_id = self.tracker.log_dataset(
            dataset_path=self.split_dir,
            dataset_name="llm_ecommerce_chatbot_sft",
            dataset_version=self.run_name,
            tags=["llm", "ecommerce", "sft", "qlora"],
        )
        self.tracker.report_table(
            "llm_dataset/train_preview",
            "formatted_text",
            train_dataset.select(range(min(20, len(train_dataset)))).to_pandas(),
        )
        self.tracker.report_table(
            "llm_dataset/eval_preview",
            "formatted_text",
            eval_dataset.select(range(min(20, len(eval_dataset)))).to_pandas(),
        )

        return {"train": train_dataset, "eval": eval_dataset}

    @staticmethod
    def _first_text_value(example: dict[str, Any], keys: tuple[str, ...]) -> str:
        """Return the first non-empty text value from a dataset row.

        Args:
            example: Dataset row.
            keys: Candidate text column names.

        Returns:
            Text value.
        """
        for key in keys:
            value = example.get(key)
            if value is not None and str(value).strip():
                return str(value)
        raise ValueError(f"None of the expected text columns were present: {keys}")

    def train(self) -> str:
        """Execute the QLoRA fine-tuning pipeline.

        Returns:
            The absolute path to the saved LoRA adapter.
        """
        print(f"Starting QLoRA fine-tuning for {self.base_model_id}")

        hf_token_kwargs = self._hf_token_kwargs()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                **hf_token_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(self._base_model_load_error()) from exc
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        datasets = self._format_dataset(tokenizer=tokenizer)

        model_kwargs: dict[str, Any] = dict(hf_token_kwargs)
        is_quantized = False
        pre_quantized = "4bit" in self.base_model_id.lower()
        if torch.cuda.is_available() and not pre_quantized:
            compute_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            model_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=compute_dtype,
                    ),
                    "device_map": "auto",
                }
            )
            is_quantized = True
        elif torch.cuda.is_available() and pre_quantized:
            model_kwargs["device_map"] = "auto"
            is_quantized = True
        else:
            print(
                "CUDA is not available; loading the base model without 4-bit "
                "quantization. Use a small base model for CPU runs."
            )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                **model_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(self._base_model_load_error()) from exc
        model.config.use_cache = False
        if is_quantized:
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=0.1,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self._target_modules(model),
        )

        final_output_dir = self.output_dir / (
            self.task.id if self.task else self.run_name
        )
        cuda_available = torch.cuda.is_available()
        bf16_enabled = cuda_available and torch.cuda.is_bf16_supported()
        fp16_enabled = cuda_available and not bf16_enabled

        training_args = SFTConfig(
            output_dir=str(final_output_dir),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit" if is_quantized else "adamw_torch",
            save_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            max_grad_norm=0.3,
            max_steps=-1,  # Automatically calculated from epochs
            num_train_epochs=self.epochs,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=100,
            report_to=["clearml"] if self.task else [],
            dataset_text_field="text",
            max_length=self.max_seq_length,
            bf16=bf16_enabled,
            fp16=fp16_enabled,
            use_cpu=not cuda_available,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            peft_config=peft_config,
            args=training_args,
            processing_class=tokenizer,
        )
        if self.task:
            trainer.add_callback(
                ClearMLTrainerMetricCallback(
                    tracker=self.tracker,
                    run_name=self.run_name,
                )
            )

        train_output = trainer.train()
        train_metrics = dict(train_output.metrics)
        eval_metrics = dict(trainer.evaluate())
        log_history = list(trainer.state.log_history)

        adapter_dir = final_output_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        train_metrics_path = final_output_dir / "train_metrics.json"
        eval_metrics_path = final_output_dir / "eval_metrics.json"
        log_history_path = final_output_dir / "trainer_log_history.json"
        adapter_archive = Path(f"{adapter_dir}.zip")
        finetuned_ollama_model = self._finetuned_ollama_model_name()
        serving_endpoint = (
            f"{CLEARML_SERVING_BASE_URL.rstrip('/')}/"
            f"{CLEARML_SERVING_ENDPOINT.strip('/')}-llm"
        )
        manifest = {
            "adapter_dir": str(adapter_dir),
            "adapter_archive": str(adapter_archive),
            "base_model_id": self.base_model_id,
            "clearml_dataset_id": self.clearml_dataset_id,
            "clearml_llm_serving_endpoint": serving_endpoint,
            "finetuned_ollama_model": finetuned_ollama_model,
            "max_seq_length": self.max_seq_length,
            "max_train_samples": self.max_train_samples,
            "ollama_model": self.ollama_model,
            "run_name": self.run_name,
            "train_dataset_size": len(datasets["train"]),
            "eval_dataset_size": len(datasets["eval"]),
        }

        save_json(train_metrics_path, train_metrics)
        save_json(eval_metrics_path, eval_metrics)
        save_json(log_history_path, log_history)
        save_json(final_output_dir / "fine_tune_manifest.json", manifest)
        save_json(adapter_dir / "fine_tune_manifest.json", manifest)
        adapter_archive = Path(
            shutil.make_archive(str(adapter_dir), "zip", adapter_dir)
        )
        save_json(self.output_dir / "latest_finetune.json", manifest)

        self._report_trainer_history(log_history)
        self.tracker.report_metrics(
            "llm_finetuning/train", self.run_name, train_metrics
        )
        self.tracker.report_metrics("llm_finetuning/eval", self.run_name, eval_metrics)
        self.tracker.report_table(
            "llm_finetuning/trainer_log_history",
            self.run_name,
            self._history_to_dataframe(log_history),
        )
        self.tracker.upload_artifacts(
            {
                "lora_adapter_archive": adapter_archive,
                "train_metrics": train_metrics_path,
                "eval_metrics": eval_metrics_path,
                "trainer_log_history": log_history_path,
                "fine_tune_manifest": final_output_dir / "fine_tune_manifest.json",
            }
        )
        self.clearml_model_id = self.tracker.register_output_model(
            model_path=adapter_archive,
            model_name=f"llm-lora-{self.run_name}",
            metadata=manifest,
            framework="pytorch",
        )
        if self.clearml_model_id:
            manifest["clearml_model_id"] = self.clearml_model_id
            save_json(final_output_dir / "fine_tune_manifest.json", manifest)
            save_json(self.output_dir / "latest_finetune.json", manifest)
            self.tracker.upload_artifact(
                "fine_tune_manifest_with_model_id",
                final_output_dir / "fine_tune_manifest.json",
            )

        print(f"Fine-tuning complete. Adapter saved to {adapter_dir}")
        self.tracker.close()
        return str(adapter_dir)

    @staticmethod
    def _hf_token_kwargs() -> dict[str, str]:
        """Build Hugging Face token kwargs without exposing token values.

        Returns:
            Keyword arguments for Transformers model/tokenizer loading.
        """
        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        return {"token": token} if token else {}

    def _base_model_load_error(self) -> str:
        """Build an actionable model-loading error message.

        Returns:
            Error message.
        """
        return (
            f"Could not load base model {self.base_model_id!r}. "
            "For gated Hugging Face models, accept the model license and set "
            "HF_TOKEN/HUGGINGFACE_TOKEN, or run `huggingface-cli login`. "
            "For a no-token local smoke run use: "
            "make finetune-ollama FINETUNE_OLLAMA_MODEL=tinyllama:1.1b "
            "FINETUNE_MAX_TRAIN_SAMPLES=200 FINETUNE_EPOCHS=1"
        )

    def _finetuned_ollama_model_name(self) -> str | None:
        """Return the Ollama model name produced by the adapter serving script.

        Returns:
            Finetuned Ollama model name, if an Ollama model tag is configured.
        """
        if not self.ollama_model:
            return None
        return f"finetuned-{self.ollama_model.replace(':', '_')}"

    def _report_trainer_history(self, log_history: list[dict[str, Any]]) -> None:
        """Report Trainer log history as ClearML scalar curves.

        Args:
            log_history: Hugging Face Trainer log history.
        """
        metric_points: dict[str, list[tuple[int, float]]] = {}
        for index, record in enumerate(log_history):
            iteration = int(record.get("step") or index)
            for metric_name, value in record.items():
                if metric_name in {"step"}:
                    continue
                scalar_value = self.as_float(value)
                if scalar_value is None:
                    continue
                metric_points.setdefault(metric_name, []).append(
                    (iteration, scalar_value)
                )

        for metric_name, points in metric_points.items():
            self.tracker.report_scalar_points(
                f"llm_finetuning/history/{metric_name}",
                self.run_name,
                points,
            )

    @staticmethod
    def _history_to_dataframe(log_history: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert Trainer log history into a dataframe.

        Args:
            log_history: Hugging Face Trainer log history.

        Returns:
            Log history dataframe.
        """
        if not log_history:
            return pd.DataFrame([{"message": "no trainer log history"}])
        return pd.DataFrame(log_history)

    @staticmethod
    def as_float(value: Any) -> float | None:
        """Convert numeric values to floats and ignore non-scalars.

        Args:
            value: Candidate scalar value.

        Returns:
            Float value, if numeric.
        """
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except TypeError, ValueError:
            return None

    @staticmethod
    def _target_modules(model: AutoModelForCausalLM) -> list[str]:
        """Infer LoRA target modules from a loaded causal LM.

        Args:
            model: Loaded model.

        Returns:
            Module name suffixes compatible with PEFT.
        """
        module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        candidate_groups = [
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["query_key_value"],
            ["c_attn", "c_proj"],
            ["Wqkv", "out_proj"],
        ]
        for candidates in candidate_groups:
            if any(candidate in module_names for candidate in candidates):
                return [
                    candidate for candidate in candidates if candidate in module_names
                ]
        raise ValueError(
            "Could not infer LoRA target modules for this model. "
            "Use a Llama/Mistral-style causal LM or extend _target_modules."
        )
