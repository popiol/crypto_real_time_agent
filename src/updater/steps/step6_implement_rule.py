"""Implement rule — design.md §8.2 Step 6.

Generates real, executable Python code from the idea step5_generate_idea.py's
step persisted to data/state/rule_idea.json (new_rule → new folder/v1.py;
fix → new version alongside the existing one), validating and self-correcting
syntax errors via an LLM diff loop, then commits the new rule file to git.

run() is this pipeline.py stage's entry point: it reads rule_idea.json,
skipping if there's nothing there for the current cycle_id (either step3's
run decided action=continue and step5 never produced an idea, or a stale
file from a run where nothing consumed it — either way, not fresh),
otherwise implements it and records the outcome via plan_next_cycle.py.

Reads:
  data/state/rule_idea.json
Writes:
  src/strategy/rules/<rule_name>/v<N>.py, data/state/plan.json,
  data/state/rule_idea.json (cleared)
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
from src.agent.models import AppConfig
from src.updater import paths
from src.updater import plan_next_cycle as plan_next_cycle_step
from src.updater.code_diff import CodeDiff, apply_changes
from src.updater.llm import make_llm
from src.updater.models import ImplementedRule, PendingRuleIdea, RuleIdea
from src.updater.rule_constraints import (
    DATA_WINDOW_CONSTRAINT,
    LONG_ONLY_CONSTRAINT,
    SIGNAL_FIELDS_CONSTRAINT,
    STATELESS_CONSTRAINT,
)

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

_IMPLEMENT_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "Generate a complete, self-contained Python module that implements the described rule. "
    "Return ONLY the raw Python source code — no explanation, no markdown, no code fences. "
    + DATA_WINDOW_CONSTRAINT
    + "\n\n" + LONG_ONLY_CONSTRAINT
    + "\n\n" + SIGNAL_FIELDS_CONSTRAINT
    + "\n\n" + STATELESS_CONSTRAINT
)

_FIX_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "You will be given a partial or broken implementation of a trading rule. "
    "Return the changes needed to complete it into a fully working module. "
    "Do not rewrite parts that are already correct.\n\n"
    + DATA_WINDOW_CONSTRAINT
    + "\n\n" + LONG_ONLY_CONSTRAINT
    + "\n\n" + SIGNAL_FIELDS_CONSTRAINT
    + "\n\n" + STATELESS_CONSTRAINT
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


def run(config: AppConfig, state_dir: Path, cycle_id: str) -> None:
    pending = _load_pending_idea(state_dir, cycle_id)
    if pending is None:
        logger.info("No fresh rule idea for this cycle; skipping implementation")
        return
    idea, plan, analysis = pending.idea, pending.plan, pending.analysis

    rule_id, rule_path = next_rule_path(idea)
    implemented_rule_id: str | None = None
    try:
        implemented = generate_code(idea, rule_id, config.llm_model)
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        rule_path.write_text(implemented.code, encoding="utf-8")
        commit_and_push(rule_id, implemented.function_name)
        implemented_rule_id = rule_id
        logger.info("Implemented rule %s at %s", rule_id, rule_path)
    except ImplementationFailed:
        logger.exception("Rule implementation failed for idea '%s'", idea.title)

    if implemented_rule_id:
        # The new rule just took over and hasn't had a chance to run yet, so
        # there's nothing to evaluate — do NOT re-diagnose the old rule here.
        # That plan would only be read next cycle, by which point the active
        # rule will already be this new one, and a stale "fix" verdict about
        # its predecessor would get misapplied to it before it ever got a
        # chance to prove itself.
        plan_next_cycle_step.write_implemented(state_dir, implemented_rule_id, cycle_id)
    else:
        # Implementation failed; keep evaluating the still-active rule.
        plan_next_cycle_step.write_next_cycle_plan(state_dir, plan, analysis, cycle_id, config)

    paths.rule_idea(state_dir).unlink(missing_ok=True)


def _load_pending_idea(state_dir: Path, cycle_id: str) -> PendingRuleIdea | None:
    path = paths.rule_idea(state_dir)
    if not path.exists():
        return None
    try:
        pending = PendingRuleIdea.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read rule_idea.json", exc_info=True)
        return None
    if pending.cycle_id != cycle_id:
        logger.info(
            "rule_idea.json is from cycle %s, not the current cycle %s; skipping",
            pending.cycle_id,
            cycle_id,
        )
        return None
    return pending


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
    bad_call = _find_indicators_kwarg(tree)
    if bad_call is not None:
        return (
            f"Line {bad_call}: BuySignal/SellSignal must not be constructed with "
            "indicators= — that field is filled in automatically after signal() "
            "returns; any value set here is silently overwritten and gets validated "
            "as dict[str, float | None] in the meantime, so non-numeric values (or "
            "custom exit parameters like profit_target_pct) crash the moment the "
            "rule fires. Remove the indicators= argument entirely."
        )
    bad_attr = _find_position_state_access(tree)
    if bad_attr is not None:
        return (
            f"Line {bad_attr}: signal() has no access to open-position/portfolio "
            "state — there is no .positions/.portfolio/.holdings attribute anywhere "
            "in this system. MarketData is plain market data only (dict[str, "
            "PairData]); entry price and position bookkeeping live outside signal() "
            "and are never passed into it. Remove this access and express exit "
            "conditions using only market data (indicators, price levels, time), "
            "not a value relative to an assumed entry price."
        )
    return None


def _find_indicators_kwarg(tree: ast.AST) -> int | None:
    """Return the line number of the first BuySignal/SellSignal(... indicators=...)
    call, or None if there isn't one.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id in ("BuySignal", "SellSignal")):
            continue
        if any(kw.arg == "indicators" for kw in node.keywords):
            return node.lineno
    return None


_HALLUCINATED_POSITION_ATTRS = {"positions", "portfolio", "open_positions", "holdings"}


def _find_position_state_access(tree: ast.AST) -> int | None:
    """Return the line number of the first access to a nonexistent position/portfolio
    attribute (e.g. data.positions), or None if there isn't one.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _HALLUCINATED_POSITION_ATTRS:
            return node.lineno
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
