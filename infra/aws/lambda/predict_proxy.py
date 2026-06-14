"""Lambda proxy for SageMaker predictions."""

from __future__ import annotations

import json
import os

import boto3

runtime = boto3.client("sagemaker-runtime")


def handler(event, context):
    """Invoke the configured SageMaker endpoint.

    Args:
        event: Lambda event.
        context: Lambda runtime context.

    Returns:
        API Gateway-compatible response.
    """

    _ = context
    body = event.get("body") or "{}"
    endpoint_name = os.environ["SAGEMAKER_ENDPOINT_NAME"]
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=body,
    )
    payload = response["Body"].read().decode("utf-8")
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(json.loads(payload)),
    }
