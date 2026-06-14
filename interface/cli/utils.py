"""Shared helpers for Click command modules."""

# JSON serialization is a best-effort CLI presentation boundary.

from __future__ import annotations

import json
import sys
from typing import Any

import click

HELP_CONTEXT = {"help_option_names": ["-h", "--help"]}


def make_json_serializable(obj: Any) -> Any:
    """Convert custom objects into JSON-serializable Python types.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable representation.
    """
    if isinstance(obj, dict) or hasattr(obj, "items"):
        return {str(key): make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(value) for value in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def echo_json(result: Any) -> None:
    """Print a command result as stable JSON.

    Args:
        result: Command result payload.
    """
    click.echo(json.dumps(make_json_serializable(result), indent=2, sort_keys=True))


def run_command(command: click.Command, prog_name: str | None = None) -> None:
    """Run one Click command as a console-script entrypoint.

    Args:
        command: Click command to run.
        prog_name: Program name shown by Click.
    """
    command.main(args=sys.argv[1:], prog_name=prog_name, standalone_mode=True)
