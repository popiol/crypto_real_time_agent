"""Test the step8 implementation + fix loop.

Sends a real code-generation request to the LLM, truncates the result to
introduce a syntax error, then runs _fix_with_diff until the code compiles.
"""

from src.updater.llm import make_llm
from src.updater.models import RuleIdea
from src.updater.steps.step8_implement_idea import (
    _MAX_FIX_ATTEMPTS,
    _check_syntax,
    _fix_with_diff,
    _initial_code,
)

_MODEL = "gemini-2.5-flash"

_IDEA = RuleIdea(
    idea_id="test-001",
    title="RSI Overbought/Oversold",
    description=(
        "Emit a buy signal when the 14-period RSI drops below 30 "
        "and a sell signal when it rises above 70."
    ),
    rationale="RSI extremes often precede mean reversion.",
    pseudocode=(
        "compute RSI(14) from hot tick close prices; "
        "if RSI < 30 emit BuySignal; if RSI > 70 emit SellSignal"
    ),
)


def test_fix_loop_recovers_from_truncation() -> None:
    llm = make_llm(_MODEL)

    # Step 1: generate initial code
    code = _initial_code(_IDEA, llm)
    assert code, "LLM returned empty code"
    print(f"\nGenerated {len(code)} chars")
    print(code)

    # Step 2: truncate at midpoint — signal function will be missing or incomplete
    truncated = code[: len(code) // 2]
    print(f"\nTruncated to {len(truncated)} chars")
    print(truncated)
    error = _check_syntax(truncated)
    assert error is not None, "Truncated code should fail _check_syntax"
    print(f"Error after truncation: {error}")

    # Step 3: fix loop
    current = truncated
    for attempt in range(_MAX_FIX_ATTEMPTS):
        error = _check_syntax(current)
        if error is None:
            break
        print(f"Fix attempt {attempt + 1}: {error}")
        current = _fix_with_diff(current, _IDEA, llm)
        print(f"After fix attempt {attempt + 1}, code length: {len(current)}")
        print(current)
    else:
        error = _check_syntax(current)
        assert error is None, (
            f"Still broken after {_MAX_FIX_ATTEMPTS} attempts: {error}"
        )

    assert _check_syntax(current) is None, "Final code does not compile"
    print(f"Fixed after {attempt + 1} attempt(s), {len(current)} chars")


if __name__ == "__main__":
    test_fix_loop_recovers_from_truncation()
    print("\nPASS")
