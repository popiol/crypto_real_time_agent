"""LangChain model factory and structured-output helper.

Provider is inferred from the model name prefix:
  gemini-*   → langchain_google_genai.ChatGoogleGenerativeAI
  claude-*   → langchain_anthropic.ChatAnthropic
  gpt-* / o* → langchain_openai.ChatOpenAI

Install the matching provider package alongside langchain-core:
  pip install langchain-core langchain-google-genai   # for Gemini
  pip install langchain-core langchain-anthropic      # for Claude
  pip install langchain-core langchain-openai         # for OpenAI
"""

from __future__ import annotations

import logging
from typing import TypeVar

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

load_dotenv(".env")

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_TOKENS = 4096


def make_llm(model: str) -> BaseChatModel:
    if model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, max_tokens=_MAX_TOKENS)
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, max_tokens=_MAX_TOKENS)
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, max_tokens=_MAX_TOKENS)
    raise ValueError(
        f"Cannot infer LangChain provider from model name '{model}'. "
        "Expected prefix: gemini-, claude-, gpt-, o1, o3, o4."
    )


def llm_structured(model: str, system: str, user: str, output_type: type[T]) -> T:
    """Call the LLM and parse the response into output_type."""
    llm = make_llm(model)
    structured = llm.with_structured_output(output_type)
    result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    assert isinstance(result, output_type), (
        f"Expected {output_type.__name__}, got {type(result).__name__}"
    )
    return result
