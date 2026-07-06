"""Test that the Gemini model can generate long Python code without truncation."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.updater.llm import make_llm


def test_long_code_generation() -> None:
    llm = make_llm("gemini-2.5-flash")
    response = llm.invoke([
        SystemMessage(content="You are an expert Python developer."),
        HumanMessage(content=(
            "Write a complete, self-contained Python module that implements a "
            "technical-analysis helper library. Include at least:\n"
            "- RSI calculation\n"
            "- Bollinger Bands calculation\n"
            "- MACD calculation\n"
            "- A simple moving average\n"
            "- An exponential moving average\n"
            "Each function must have a full implementation with type hints and "
            "inline comments explaining the maths. Aim for roughly 5000 characters "
            "of source code. Return ONLY the raw Python source — no markdown fences."
        )),
    ])
    raw = response.content
    code = raw if isinstance(raw, str) else "".join(
        p if isinstance(p, str) else p.get("text", "") for p in raw
    )
    print(f"\nGenerated {len(code)} characters")
    print("--- first 200 chars ---")
    print(code[:200])
    print("--- last 200 chars ---")
    print(code[-200:])
    assert len(code) >= 3000, f"Output too short: {len(code)} chars (expected >= 3000)"


if __name__ == "__main__":
    test_long_code_generation()
    print("\nPASS")
