"""Azure Function proxy for Azure ML predictions."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Invoke the configured Azure ML endpoint.

    Args:
        req: HTTP request.

    Returns:
        HTTP response.
    """

    try:
        payload = req.get_json()
    except ValueError:
        return _json_response({"detail": "Request body must be valid JSON."}, 400)

    scoring_uri = os.getenv("AZURE_ML_SCORING_URI", "").strip()
    if not scoring_uri:
        return _json_response(
            {
                "detail": "AZURE_ML_SCORING_URI is not configured.",
                "endpoint": os.getenv("AZURE_ML_ENDPOINT_NAME", ""),
            },
            503,
        )

    try:
        status_code, response_payload = _invoke_azure_ml(
            scoring_uri=scoring_uri,
            payload=payload,
            api_key=os.getenv("AZURE_ML_API_KEY", ""),
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") or exc.reason
        return _json_response({"detail": detail}, exc.code)
    except urllib.error.URLError as exc:
        return _json_response({"detail": str(exc.reason)}, 502)

    return _json_response(response_payload, status_code)


def _invoke_azure_ml(scoring_uri: str, payload: Any, api_key: str) -> tuple[int, Any]:
    """Invoke the configured Azure ML scoring endpoint.

    Args:
        scoring_uri: Azure ML endpoint scoring URI.
        payload: Prediction payload.
        api_key: Optional Azure ML endpoint key.

    Returns:
        HTTP status code and decoded response payload.
    """

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        scoring_uri,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw_payload = response.read().decode("utf-8")
        return response.status, _decode_response(raw_payload)


def _decode_response(raw_payload: str) -> Any:
    """Decode an Azure ML response payload.

    Args:
        raw_payload: Raw response string.

    Returns:
        JSON-decoded payload when possible, otherwise a raw wrapper.
    """

    if not raw_payload:
        return {}
    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError:
        return {"raw": raw_payload}


def _json_response(payload: Any, status_code: int) -> func.HttpResponse:
    """Build a JSON HTTP response.

    Args:
        payload: JSON-serializable payload.
        status_code: HTTP status code.

    Returns:
        Azure Function HTTP response.
    """

    return func.HttpResponse(
        json.dumps(payload),
        status_code=status_code,
        mimetype="application/json",
    )
