"""Update indicator set — design.md §8.2 Step 1. Updates or bootstraps the
LLM-defined indicator set.

Skipped when plan.json has action='continue' and indicator_set.json already
exists (previous cycle succeeded; no need to change what we measure).

On first run (no indicator_set.json) or on failure cycles (action='fix' or
'new_rule'), the process is two-phase:
  1. LLM defines the list of indicators (names and descriptions only).
  2. For each indicator, LLM generates the compute() function individually.

Each indicator is a Python function with signature:
    def compute(data: PairData) -> float | None

Writes: data/state/indicator_set.json
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

import src.agent.models as _agent_models
from src.agent.models import AppConfig
from src.updater import paths
from src.updater.code_diff import CodeDiff, apply_changes
from src.updater.llm import llm_structured, make_llm
from src.updater.models import Indicator, IndicatorSet, Plan

logger = logging.getLogger(__name__)

_MODELS_SOURCE = "\n\n".join(
    inspect.getsource(cls)
    for cls in (
        _agent_models.Tick,
        _agent_models.WarmCandle,
        _agent_models.ColdMonth,
        _agent_models.PairData,
    )
)

_EXAMPLE_INDICATOR = """\
def compute(data: PairData) -> float | None:
    \"\"\"Return the ratio of current bid-ask spread to its 10-tick average.\"\"\"
    ticks = data.hot
    if len(ticks) < 10:
        return None
    avg = sum(t.spread_rel for t in ticks[-10:]) / 10
    if avg == 0:
        return None
    return ticks[-1].spread_rel / avg
"""

_DEFINE_SYSTEM = (
    "You are a quantitative trading analyst. "
    "Define a set of indicators suitable for detecting mean-reversion opportunities "
    "in cryptocurrency markets using hot (tick), warm (hourly candle), and cold (monthly) data. "
    "For each indicator provide only its name and a one-sentence description of what it measures. "
    "Do NOT write any code."
)

_CODE_SYSTEM = (
    "You are an expert Python developer. "
    "Implement the described indicator as a Python function with this exact signature: "
    "`def compute(data: PairData) -> float | None`. "
    "Use only the Python standard library and the provided PairData model — no external packages. "
    "Return None when there is insufficient data. "
    "Return ONLY the raw Python function — no explanation, no markdown, no code fences."
)

_FIX_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading indicators. "
    "You will be given a partial or broken implementation of an indicator function. "
    "Return the changes needed to complete it into a fully working function. "
    "Do not rewrite parts that are already correct."
)

_MAX_FIX_ATTEMPTS = 5


class _IndicatorSpec(BaseModel):
    indicator_id: str
    name: str
    description: str


class _IndicatorSpecList(BaseModel):
    indicators: list[_IndicatorSpec]


def run(config: AppConfig, state_dir: Path, cycle_id: str) -> None:
    set_path = paths.indicator_set(state_dir)
    plan_path = paths.plan(state_dir)

    bootstrap = not set_path.exists()
    if bootstrap:
        action, plan_description = "new_rule", None
        logger.info("No indicator_set.json found; bootstrapping initial indicator set")
    else:
        action, plan_description = _read_plan(plan_path)
        if action == "continue":
            logger.info("Indicator set unchanged (previous cycle succeeded)")
            return
        logger.info(
            "Revising indicator set: plan action=%s — %s",
            action, plan_description or "(no description)",
        )

    current: IndicatorSet | None = None
    if not bootstrap:
        try:
            current = IndicatorSet.model_validate_json(
                set_path.read_text(encoding="utf-8")
            )
        except Exception:
            logger.warning(
                "Could not parse existing indicator_set.json; will regenerate"
            )

    updated = _llm_update(current, plan_description, config.llm_model)
    set_path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "indicator_set.json written: %d indicator(s), version=%s",
        len(updated.indicators),
        updated.version,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _read_plan(plan_path: Path) -> tuple[str, str | None]:
    if not plan_path.exists():
        return "new_rule", None
    try:
        plan = Plan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        return plan.action, plan.description
    except Exception:
        logger.warning("Could not parse plan.json; treating as new_rule")
        return "new_rule", None


def _llm_update(
    current: IndicatorSet | None,
    plan_description: str | None,
    model: str,
) -> IndicatorSet:
    # Phase 1: define indicator specs (names + descriptions only)
    try:
        specs = _define_specs(current, plan_description, model)
    except Exception:
        logger.exception("LLM call to define indicator specs failed")
        if current is not None:
            logger.warning("Retaining existing indicator set unchanged")
            return current
        raise

    # Phase 2: generate code for each spec individually
    indicators: list[Indicator] = []
    for spec in specs:
        try:
            code = _generate_code(spec, model)
            indicators.append(
                Indicator(
                    indicator_id=spec.indicator_id,
                    name=spec.name,
                    description=spec.description,
                    code=code,
                )
            )
        except Exception:
            logger.warning(
                "Failed to generate code for indicator '%s'; skipping",
                spec.name,
                exc_info=True,
            )

    if not indicators and current is not None:
        logger.warning("No valid indicators generated; retaining existing set")
        return current

    return IndicatorSet(
        version=str(uuid.uuid4()),
        indicators=indicators,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _define_specs(
    current: IndicatorSet | None,
    plan_description: str | None,
    model: str,
) -> list[_IndicatorSpec]:
    if current is None:
        user = (
            "Define 5–8 diverse indicators covering: "
            "momentum, volatility, spread, volume, and price extremes.\n\n"
            "Available data tiers:\n"
            "  hot  — recent ticks (last ~5 min)\n"
            "  warm — last 24 hourly candles\n"
            "  cold — monthly stats\n\n"
            "For each indicator provide indicator_id (snake_case slug), name, and description only."
        )
    else:
        current_summary = json.dumps(
            [
                {
                    "indicator_id": i.indicator_id,
                    "name": i.name,
                    "description": i.description,
                }
                for i in current.indicators
            ],
            indent=2,
        )
        plan_section = (
            f"\nCycle plan:\n{plan_description}\n" if plan_description else ""
        )
        user = (
            f"Current indicators:\n{current_summary}\n"
            f"{plan_section}\n"
            "Review the list. You may add, remove, or modify indicators based on the plan. "
            "Return the complete updated list (including unchanged entries) with "
            "indicator_id, name, and description only — no code."
        )

    result = llm_structured(
        model=model, system=_DEFINE_SYSTEM, user=user, output_type=_IndicatorSpecList
    )
    return result.indicators


def _generate_code(spec: _IndicatorSpec, model: str) -> str:
    base_context = (
        f"Indicator name: {spec.name}\n"
        f"Description: {spec.description}\n\n"
        f"Available data models:\n{_MODELS_SOURCE}\n\n"
        f"Example:\n{_EXAMPLE_INDICATOR}"
    )
    llm = make_llm(model)

    response = llm.invoke(
        [SystemMessage(content=_CODE_SYSTEM), HumanMessage(content=base_context)]
    )
    code = _extract_code(response.content)
    logger.debug("Initial code for indicator '%s' (%d chars)", spec.name, len(code))

    for attempt in range(_MAX_FIX_ATTEMPTS):
        error = _check_syntax(code)
        if error is None:
            logger.debug(
                "Indicator '%s' passed validation after %d fix attempt(s)",
                spec.name,
                attempt,
            )
            return code
        logger.warning(
            "Indicator '%s' validation error (attempt %d/%d): %s\nCode:\n%s",
            spec.name,
            attempt + 1,
            _MAX_FIX_ATTEMPTS,
            error,
            code,
        )
        code = _fix_with_diff(code, base_context, llm)

    error = _check_syntax(code)
    if error is not None:
        logger.error(
            "Indicator '%s' still invalid after %d fix attempts: %s\nCode:\n%s",
            spec.name,
            _MAX_FIX_ATTEMPTS,
            error,
            code,
        )
        raise ValueError(
            f"Indicator '{spec.name}' still invalid after {_MAX_FIX_ATTEMPTS} fix attempts: {error}"
        )
    return code


def _fix_with_diff(code: str, base_context: str, llm) -> str:
    structured = llm.with_structured_output(CodeDiff)
    user_prompt = (
        f"{base_context}\n\n"
        f"Entry-point function name: compute\n\n"
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


def _extract_code(raw: object) -> str:
    text = (
        raw
        if isinstance(raw, str)
        else "".join(p if isinstance(p, str) else p.get("text", "") for p in raw)
    ).strip()
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _check_syntax(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "compute"
        ),
        None,
    )
    if fn is None:
        return "Missing required function: compute(data: PairData) -> float | None"
    if not any(isinstance(n, ast.Return) for n in ast.walk(fn)):
        return "compute() has no return statement — function body is likely incomplete"
    return None
