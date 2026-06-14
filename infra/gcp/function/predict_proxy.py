"""Google Cloud Function proxy for Vertex AI predictions."""

from __future__ import annotations

import os

from google.cloud import aiplatform


def handler(request):
    """Invoke the configured Vertex AI endpoint.

    Args:
        request: Flask-compatible request.

    Returns:
        Prediction response.
    """

    project_id = os.environ["GCP_PROJECT_ID"]
    location = os.environ["GCP_LOCATION"]
    endpoint_id = os.environ["VERTEX_ENDPOINT_ID"]
    aiplatform.init(project=project_id, location=location)
    endpoint = aiplatform.Endpoint(endpoint_id)
    payload = request.get_json(silent=True) or {}
    prediction = endpoint.predict(instances=[payload])
    return {"predictions": prediction.predictions}
