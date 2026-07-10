"""Claude Code CLI — non-interactive LLM calls.

Wraps the local `claude` binary. Requires the binary to be on PATH and
authenticated via OAuth (does NOT need ANTHROPIC_API_KEY — do not use --bare).
"""

from __future__ import annotations

import json
import subprocess
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_TIMEOUT = 120


def cli_call(system: str, user: str) -> str:
    """Run a plain-text prompt through the Claude CLI and return the response."""
    return _run(system=system, user=user)


def cli_call_structured(system: str, user: str, output_type: type[T]) -> T:
    """Run a structured prompt through the Claude CLI and parse into output_type."""
    schema = json.dumps(output_type.model_json_schema())
    raw = _run(system=system, user=user, json_schema=schema)
    return output_type.model_validate_json(raw)


def _run(system: str, user: str, json_schema: str | None = None) -> str:
    cmd = [
        "claude",
        "--print",
        "--no-session-persistence",
        "--tools", "",
    ]
    if system:
        cmd += ["--system-prompt", system]
    if json_schema:
        cmd += ["--output-format", "json", "--json-schema", json_schema]
    cmd.append(user)

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", timeout=_TIMEOUT
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"claude CLI exited {result.returncode}: {stderr}")

    stdout = (result.stdout or "").strip()
    if not stdout:
        raise RuntimeError("claude CLI returned empty output")

    if json_schema:
        envelope = json.loads(stdout)
        payload = envelope.get("result", stdout)
        # result field may be a JSON string or an already-decoded dict
        return payload if isinstance(payload, str) else json.dumps(payload)

    return stdout
