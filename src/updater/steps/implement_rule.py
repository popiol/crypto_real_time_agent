"""Implement rule — design.md §8.2 Step 5.

Generates real, executable Python code from the idea produced by
generate_idea.py (new_rule → new folder/v1.py; fix → new version alongside
the existing one), validating and self-correcting syntax errors via an LLM
diff loop, then commits the new rule file to git.

Not a pipeline.py stage in its own right: it consumes the RuleIdea produced
in-memory by generate_idea.py in the same run, so this module is called
directly from step7_implement_idea.py's orchestrator.

Writes: src/strategy/rules/<rule_name>/v<N>.py
"""

from __future__ import annotations

import ast
import inspect
import logging
import re
import subprocess
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

import src.agent.models as _agent_models
from src.updater.code_diff import CodeDiff, apply_changes
from src.updater.llm import make_llm
from src.updater.models import ImplementedRule, RuleIdea

logger = logging.getLogger(__name__)

_MODELS_SOURCE = (
    "\n\n".join(
        inspect.getsource(cls)
        for cls in (
            _agent_models.Tick,
            _agent_models.WarmCandle,
            _agent_models.ColdMonth,
            _agent_models.PairData,
            _agent_models.BuySignal,
            _agent_models.SellSignal,
        )
    )
    + "\n\nMarketData = dict[str, PairData]"
)

RULES_DIR = Path("src/strategy/rules")

_MAX_FIX_ATTEMPTS = 10

DATA_WINDOW_CONSTRAINT = (
    "DATA TIER LIMITS (hard constraints — a rule that violates these will never "
    "produce a signal against real data, even though it will look syntactically valid):\n"
    "- data.hot: the last ~300 raw ticks (~5 minutes of history at 1 poll/sec). "
    "Tick-level last_price/bid/ask/spread.\n"
    "- data.warm: the last 24 hourly OHLC candles, AT MOST — never more than 24 entries. "
    "Any indicator whose lookback exceeds 24 hourly candles (e.g. SMA(30), SMA(50), or "
    "any period > 23) CANNOT be computed from data.warm; len(data.warm) will never reach "
    "that many entries, so the rule will always fall through its own 'insufficient data' "
    "check and return [].\n"
    "- data.cold: ONE ROW PER CALENDAR MONTH, aggregates only (min_price, max_price, "
    "avg_price, avg_daily_spread, candle_count, last_candle_hour) — it is NOT an hourly "
    "or daily price series. It cannot be used to extend a warm-tier indicator's lookback "
    "(there is no way to reconstruct individual hourly/daily closes from it). Only use "
    "data.cold for coarse, monthly-resolution comparisons (e.g. current price vs. this "
    "month's avg_price/min_price/max_price).\n"
    "Every indicator lookback period MUST fit within data.warm's 24-candle limit (or use "
    "data.hot/data.cold directly). Do not design an indicator assuming a longer hourly or "
    "daily history exists anywhere — it does not."
)

_IMPLEMENT_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "Generate a complete, self-contained Python module that implements the described rule. "
    "Return ONLY the raw Python source code — no explanation, no markdown, no code fences. "
    + DATA_WINDOW_CONSTRAINT
)

_FIX_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "You will be given a partial or broken implementation of a trading rule. "
    "Return the changes needed to complete it into a fully working module. "
    "Do not rewrite parts that are already correct.\n\n"
    + DATA_WINDOW_CONSTRAINT
)

_REFERENCE_RULE = """\
\"\"\"Rule 01 — Spread compression spike (v1).\"\"\"
from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

MIN_TICKS = 10
COMPRESSION_THRESHOLD = 0.30

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS:
            continue
        spreads = [t.spread_rel for t in ticks]
        baseline = statistics.mean(spreads[:-1])
        current = spreads[-1]
        if baseline > 0 and current < baseline * (1 - COMPRESSION_THRESHOLD):
            signals.append(BuySignal(
                pair=pair,
                timestamp=ticks[-1].polled_at,
                price=ticks[-1].last_price,
            ))
        elif baseline > 0 and current > baseline * (1 + COMPRESSION_THRESHOLD):
            signals.append(SellSignal(
                pair=pair,
                timestamp=ticks[-1].polled_at,
                price=ticks[-1].last_price,
            ))
    return signals
"""


class ImplementationFailed(Exception):
    pass


def next_rule_path(idea: RuleIdea) -> tuple[str, Path]:
    """Return (versioned_rule_id, path_to_new_file)."""
    if idea.kind == "modify_rule" and idea.target_rule:
        base = re.sub(r"_v\d+$", "", idea.target_rule)
        folder = RULES_DIR / base
        existing = sorted(folder.glob("v*.py")) if folder.exists() else []
        if existing:
            m = re.match(r"v(\d+)\.py$", existing[-1].name)
            next_ver = (int(m.group(1)) + 1) if m else 2
        else:
            next_ver = 1
        rule_id = f"{base}_v{next_ver}"
        return rule_id, folder / f"v{next_ver}.py"
    else:
        existing_nums = [
            int(m.group(1))
            for d in RULES_DIR.iterdir()
            if d.is_dir() and (m := re.match(r"rule_(\d+)_", d.name))
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        slug = re.sub(r"[^a-z0-9]+", "_", idea.title.lower()).strip("_")[:30]
        folder_name = f"rule_{next_num:02d}_{slug}"
        rule_id = f"{folder_name}_v1"
        return rule_id, RULES_DIR / folder_name / "v1.py"


def _check_syntax(code: str) -> str | None:
    """Return an error description, or None if the code is a valid complete rule."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"
    signal_fn = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "signal"
        ),
        None,
    )
    if signal_fn is None:
        return "Missing required top-level function: signal(data: MarketData)"
    has_return = any(isinstance(node, ast.Return) for node in ast.walk(signal_fn))
    if not has_return:
        return "signal() has no return statement — function body is likely incomplete"
    return None


def _load_target_source(target_rule: str) -> str | None:
    """Return the source of the rule being modified, or None if unavailable."""
    m = re.match(r"^(.+)_(v\d+)$", target_rule)
    if not m:
        return None
    path = RULES_DIR / m.group(1) / f"{m.group(2)}.py"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _initial_code(idea: RuleIdea, llm: BaseChatModel) -> str:
    """First-pass code generation via plain-text LLM call."""
    existing_source = (
        _load_target_source(idea.target_rule)
        if idea.kind == "modify_rule" and idea.target_rule
        else None
    )
    existing_section = (
        f"Existing implementation to improve upon:\n{existing_source}\n\n"
        if existing_source
        else ""
    )
    user_prompt = (
        f"Implement this trading rule idea as a Python module.\n\n"
        f"Idea:\n{idea.model_dump_json(indent=2)}\n\n"
        f"{existing_section}"
        f"Available data models:\n{_MODELS_SOURCE}\n\n"
        f"Reference rule structure to follow exactly:\n{_REFERENCE_RULE}\n\n"
        "Requirements:\n"
        "1. The public entry-point function MUST be named exactly `signal` "
        "   with signature: def signal(data: MarketData) -> list[BuySignal | SellSignal].\n"
        "2. Available external packages: numpy, tensorflow, keras.\n"
        "3. Handle insufficient data gracefully (return [])."
    )
    response = llm.invoke(
        [SystemMessage(content=_IMPLEMENT_SYSTEM), HumanMessage(content=user_prompt)]
    )
    raw = response.content
    code = (
        raw
        if isinstance(raw, str)
        else "".join(p if isinstance(p, str) else p.get("text", "") for p in raw)
    ).strip()
    m = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    if m:
        code = m.group(1).strip()
    return code


def _fix_with_diff(code: str, idea: RuleIdea, llm: BaseChatModel) -> str:
    """Ask the LLM for a diff to complete/fix `code`, then apply it."""
    structured = llm.with_structured_output(CodeDiff)
    user_prompt = (
        f"Idea:\n{idea.model_dump_json(indent=2)}\n\n"
        f"Available data models:\n{_MODELS_SOURCE}\n\n"
        f"Reference rule structure:\n{_REFERENCE_RULE}\n\n"
        "Entry-point function name: signal\n\n"
        f"Partial implementation to complete:\n{code}"
    )
    try:
        result = structured.invoke(
            [SystemMessage(content=_FIX_SYSTEM), HumanMessage(content=user_prompt)]
        )
        if not isinstance(result, CodeDiff):
            raise TypeError(f"Expected CodeDiff, got {type(result).__name__}")
        logger.debug("Applying %d change(s) from diff", len(result.changes))
        return apply_changes(code, result.changes)
    except Exception:
        logger.warning(
            "Diff generation/application failed; code unchanged", exc_info=True
        )
        return code


def generate_code(idea: RuleIdea, rule_id: str, model: str) -> ImplementedRule:
    llm = make_llm(model)

    logger.info("Generating initial code for idea '%s' (%s)", idea.title, idea.idea_id)
    code = _initial_code(idea, llm)
    logger.debug("Initial code (%d chars):\n%s", len(code), code)

    for attempt in range(_MAX_FIX_ATTEMPTS):
        error = _check_syntax(code)
        if error is None:
            logger.info("Code passed validation after %d fix attempt(s)", attempt)
            break
        logger.warning(
            "Validation error (attempt %d/%d): %s\nCode:\n%s",
            attempt + 1,
            _MAX_FIX_ATTEMPTS,
            error,
            code,
        )
        code = _fix_with_diff(code, idea, llm)
        logger.debug(
            "Code after fix attempt %d (%d chars):\n%s", attempt + 1, len(code), code
        )
    else:
        error = _check_syntax(code)
        if error is not None:
            logger.error(
                "Code still failing after %d fix attempts: %s\nCode:\n%s",
                _MAX_FIX_ATTEMPTS,
                error,
                code,
            )
            raise ImplementationFailed(
                f"Code still failing after {_MAX_FIX_ATTEMPTS} fix attempts: {error}"
            )

    return ImplementedRule(
        idea_id=idea.idea_id,
        rule_id=rule_id,
        function_name="signal",
        code=code,
    )


def commit_and_push(rule_id: str, function_name: str) -> None:
    commit_msg = f"agent: implement rule {rule_id} ({function_name})"
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "--set-upstream", "origin", "HEAD"], check=True)
        logger.info("Committed and pushed: %s", commit_msg)
    except subprocess.CalledProcessError:
        logger.exception("git commit/push failed after implementing %s", rule_id)
