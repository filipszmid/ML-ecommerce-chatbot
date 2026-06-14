"""ClearML Serving proxy endpoint for the finetuned Ollama chat model."""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


class Preprocess:
    """Proxy ClearML Serving requests to an Ollama chat model."""

    def load(self, local_model_file: str) -> dict[str, Any]:
        """Load the ClearML model manifest.

        Args:
            local_model_file: Local ClearML model artifact path.

        Returns:
            Loaded manifest payload.
        """

        manifest = self._load_manifest(Path(local_model_file))
        self._ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://ollama:11434",
        ).rstrip("/")
        self._model_name = (
            manifest.get("finetuned_ollama_model")
            or os.getenv("CLEARML_LLM_OLLAMA_MODEL")
            or os.getenv("OLLAMA_MODEL")
        )
        if not self._model_name:
            raise ValueError(
                "No Ollama model configured. Set CLEARML_LLM_OLLAMA_MODEL or "
                "register a manifest with finetuned_ollama_model."
            )
        return manifest

    def preprocess(
        self, request: dict[str, Any], state: dict[str, Any], _
    ) -> dict[str, Any]:
        """Normalize the incoming chat request.

        Args:
            request: Incoming request body.
            state: Request state.
            _: Optional custom statistics callback.

        Returns:
            Ollama-compatible request payload.
        """

        messages = request.get("messages")
        if not messages:
            prompt = (
                request.get("prompt") or request.get("message") or request.get("text")
            )
            if not prompt:
                raise ValueError(
                    "Request must include messages, prompt, message, or text."
                )
            messages = [{"role": "user", "content": str(prompt)}]
        payload = {
            "model": request.get("model") or self._model_name,
            "messages": messages,
            "stream": False,
        }
        state["ollama_model"] = payload["model"]
        return payload

    def process(self, data: dict[str, Any], state: dict[str, Any], _) -> dict[str, Any]:
        """Call Ollama.

        Args:
            data: Ollama-compatible payload.
            state: Request state.
            _: Optional custom statistics callback.

        Returns:
            Chat response payload.
        """

        request = urllib.request.Request(
            url=f"{self._ollama_base_url}/api/chat",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = json.loads(response.read().decode("utf-8"))
        return {
            "model": state.get("ollama_model"),
            "response": raw.get("message", {}).get("content", ""),
            "raw": raw,
        }

    def postprocess(
        self, data: dict[str, Any], state: dict[str, Any], _
    ) -> dict[str, Any]:
        """Return the response payload.

        Args:
            data: Chat response payload.
            state: Request state.
            _: Optional custom statistics callback.

        Returns:
            Response payload.
        """

        return data

    @staticmethod
    def _load_manifest(local_model_file: Path) -> dict[str, Any]:
        """Load a manifest from a file, zip, or adapter directory.

        Args:
            local_model_file: Model artifact path.

        Returns:
            Manifest dictionary.
        """

        if local_model_file.suffix == ".zip":
            with zipfile.ZipFile(local_model_file) as archive:
                for name in archive.namelist():
                    if name.endswith("fine_tune_manifest.json"):
                        return json.loads(archive.read(name).decode("utf-8"))
            return {}
        if local_model_file.is_file() and local_model_file.suffix == ".json":
            return json.loads(local_model_file.read_text())
        manifest_path = local_model_file / "fine_tune_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {}
