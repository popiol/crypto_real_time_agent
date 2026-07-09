"""LangChain model factory and structured-output helper.

Provider is inferred from the model name prefix:
  gemini-*    → langchain_google_genai.ChatGoogleGenerativeAI
  claude-*    → langchain_anthropic.ChatAnthropic
  gpt-* / o*  → langchain_openai.ChatOpenAI
  claude-cli  → Claude Code CLI (no API key; uses local auth)

Install the matching provider package alongside langchain-core:
  pip install langchain-core langchain-google-genai   # for Gemini
  pip install langchain-core langchain-anthropic      # for Claude
  pip install langchain-core langchain-openai         # for OpenAI

For claude-cli the `claude` binary must be on PATH and authenticated.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Generic, TypeVar

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

load_dotenv(".env")

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_TOKENS = 16384
_REQUEST_TIMEOUT = 120


# ── Claude CLI wrapper ────────────────────────────────────────────────────────

class _CliResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _ClaudeCliStructured(Generic[T]):
    def __init__(self, output_type: type[T]) -> None:
        self._type = output_type

    def invoke(self, messages: list) -> T:
        system, user = _extract_messages(messages)
        schema = json.dumps(self._type.model_json_schema())
        raw = _cli_call(system=system, user=user, json_schema=schema)
        return self._type.model_validate_json(raw)


class _ClaudeCli:
    """Duck-typed wrapper around the `claude` CLI binary."""

    def invoke(self, messages: list) -> _CliResponse:
        system, user = _extract_messages(messages)
        return _CliResponse(content=_cli_call(system=system, user=user))

    def with_structured_output(self, output_type: type[T]) -> _ClaudeCliStructured[T]:
        return _ClaudeCliStructured(output_type)


def _extract_messages(messages: list) -> tuple[str, str]:
    system = ""
    user = ""
    for m in messages:
        if isinstance(m, SystemMessage):
            system = m.content if isinstance(m.content, str) else ""
        elif isinstance(m, HumanMessage):
            user = m.content if isinstance(m.content, str) else ""
    return system, user


def _cli_call(system: str, user: str, json_schema: str | None = None) -> str:
    cmd = ["claude", "--print", "--bare", "--no-session-persistence"]
    if system:
        cmd += ["--system-prompt", system]
    if json_schema:
        cmd += ["--output-format", "json", "--json-schema", json_schema]
    cmd.append(user)

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=_REQUEST_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {result.stderr.strip()}")

    stdout = result.stdout.strip()

    if json_schema:
        outer = json.loads(stdout)
        raw_result = outer.get("result", stdout)
        # result may be a string (JSON) or already a dict
        return raw_result if isinstance(raw_result, str) else json.dumps(raw_result)

    return stdout


# ── LangChain factory ─────────────────────────────────────────────────────────

def make_llm(model: str) -> BaseChatModel | _ClaudeCli:
    if model == "claude-cli":
        return _ClaudeCli()
    if model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, max_tokens=_MAX_TOKENS, timeout=_REQUEST_TIMEOUT)
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, max_tokens=_MAX_TOKENS, timeout=_REQUEST_TIMEOUT)
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, max_tokens=_MAX_TOKENS, request_timeout=_REQUEST_TIMEOUT)
    raise ValueError(
        f"Cannot infer LangChain provider from model name '{model}'. "
        "Expected prefix: gemini-, claude-, gpt-, o1, o3, o4, or 'claude-cli'."
    )


def llm_structured(model: str, system: str, user: str, output_type: type[T]) -> T:
    """Call the LLM and parse the response into output_type."""
    if model == "claude-cli":
        schema = json.dumps(output_type.model_json_schema())
        raw = _cli_call(system=system, user=user, json_schema=schema)
        result = output_type.model_validate_json(raw)
        assert isinstance(result, output_type)
        return result
    llm = make_llm(model)
    structured = llm.with_structured_output(output_type)
    result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    assert isinstance(result, output_type), (
        f"Expected {output_type.__name__}, got {type(result).__name__}"
    )
    return result
