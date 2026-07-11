"""Claude Code CLI — complete integration module.

Provides plain-text and structured LLM calls via the local `claude` binary,
plus a LangChain-compatible adapter for use with make_llm().

Requires the `claude` binary on PATH, authenticated via OAuth.
Does NOT require ANTHROPIC_API_KEY — do not add --bare, which blocks OAuth.
"""

import json
import subprocess
from typing import Generic, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_TIMEOUT = 120


# ── Core subprocess layer ─────────────────────────────────────────────────────


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
        return payload if isinstance(payload, str) else json.dumps(payload)

    return stdout


# ── LangChain-compatible adapter ──────────────────────────────────────────────


class CliResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class ClaudeCliStructured(Generic[T]):
    def __init__(self, output_type: type[T]) -> None:
        self._type = output_type

    def invoke(self, messages: list) -> T:
        system, user = _extract_messages(messages)
        return cli_call_structured(system=system, user=user, output_type=self._type)


class ClaudeCli:
    """LangChain-compatible adapter that delegates to the claude CLI binary."""

    def invoke(self, messages: list) -> CliResponse:
        system, user = _extract_messages(messages)
        return CliResponse(content=cli_call(system=system, user=user))

    def with_structured_output(self, output_type: type[T]) -> ClaudeCliStructured[T]:
        return ClaudeCliStructured(output_type)


def _extract_messages(messages: list) -> tuple[str, str]:
    system = ""
    user = ""
    for m in messages:
        if isinstance(m, SystemMessage):
            system = m.content if isinstance(m.content, str) else ""
        elif isinstance(m, HumanMessage):
            user = m.content if isinstance(m.content, str) else ""
    return system, user
