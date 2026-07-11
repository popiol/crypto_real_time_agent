"""Step 8 — Select and implement one idea.

Picks the highest-scored evaluated idea, generates the rule code via LLM,
writes the file into the versioned rule folder structure, registers it in
strategy.py, unregisters dropped/deprecated versions, and marks the idea
as implemented.

Rule folder layout:
  src/strategy/rules/<rule_name>/v1.py   ← initial version
  src/strategy/rules/<rule_name>/v2.py   ← revised version, etc.

Writes:
  - a new rule file under src/strategy/rules/<rule_name>/v<N>.py
  - updates src/strategy/strategy.py
  - updates idea_backlog.json
"""

from __future__ import annotations

import ast
import inspect
import logging
import re
import subprocess
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import src.agent.models as _agent_models
from src.agent.db import open_db
from src.agent.models import AppConfig
from src.updater.llm import make_llm
from src.updater.models import (
    IdeaBacklog,
    ImplementedRule,
    RuleEvaluation,
    RuleIdea,
    VersionComparisonResult,
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

_RULES_DIR = Path("src/strategy/rules")
_STRATEGY_FILE = Path("src/strategy/strategy.py")

_IMPLEMENT_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "Generate a complete, self-contained Python module that implements the described rule. "
    "Return ONLY the raw Python source code — no explanation, no markdown, no code fences."
)

_FIX_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "You will be given a partial or broken implementation of a trading rule. "
    "Return the changes needed to complete it into a fully working module. "
    "Do not rewrite parts that are already correct."
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

_MAX_FIX_ATTEMPTS = 10


class _ImplementationFailed(Exception):
    pass


class _CodeChange(BaseModel):
    action: Literal["add", "remove"] = Field(
        description="'add' inserts code after the given line; 'remove' deletes it."
    )
    line: int = Field(
        description="1-indexed line number. For 'add', new code is inserted after this line (0 = before line 1). For 'remove', this line is deleted."
    )
    code: str = Field(
        default="",
        description="Source text to insert. Only used for 'add'. Use \\n for multiple lines.",
    )


class _CodeDiff(BaseModel):
    changes: list[_CodeChange] = Field(
        description="Ordered list of line-level changes that together fix the syntax error."
    )


def run(config: AppConfig, state_dir: Path) -> None:
    backlog_path = state_dir / "idea_backlog.json"
    if not backlog_path.exists():
        logger.info("idea_backlog.json not found; skipping step 8")
        return

    backlog = IdeaBacklog.model_validate_json(backlog_path.read_text(encoding="utf-8"))

    # 1. Unregister dropped / deprecated versions
    _unregister_dropped(state_dir, config)

    # 2. Select best evaluated idea
    idea = _pick_best(backlog.ideas)
    if idea is None:
        logger.info("No evaluated ideas ready for implementation; skipping step 8")
        return

    # 3. Determine target file path and versioned rule_id
    rule_id, rule_path = _next_rule_path(idea)

    # 4. Generate and validate code
    try:
        implemented = _generate_code(idea, rule_id, config.llm_model)
    except _ImplementationFailed as exc:
        logger.warning("Rejecting idea %s: %s", idea.idea_id, exc)
        for i in backlog.ideas:
            if i.idea_id == idea.idea_id:
                i.status = "rejected"
                break
        backlog_path.write_text(backlog.model_dump_json(indent=2), encoding="utf-8")
        return
    except Exception:
        logger.exception("Code generation failed for idea %s", idea.idea_id)
        return

    # 5. Write rule file
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    rule_path.write_text(implemented.code, encoding="utf-8")
    logger.info("Wrote rule file: %s", rule_path)

    # 6. Register in strategy.py
    try:
        _register_rule(_STRATEGY_FILE, rule_id)
    except Exception:
        logger.exception("Failed to register %s in strategy.py", rule_id)
        rule_path.unlink(missing_ok=True)
        return

    # 7. Mark idea as implemented
    for i in backlog.ideas:
        if i.idea_id == idea.idea_id:
            i.status = "implemented"
            break
    backlog_path.write_text(backlog.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "Implemented idea '%s' as %s (%s)",
        idea.title,
        rule_id,
        implemented.function_name,
    )

    # 8. Commit and push all changes
    _commit_and_push(rule_id, implemented.function_name)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pick_best(ideas: list[RuleIdea]) -> RuleIdea | None:
    candidates: list[RuleIdea] = [
        i for i in ideas if i.status == "evaluated" and i.score is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda i: i.score or 0.0)


def _next_rule_path(idea: RuleIdea) -> tuple[str, Path]:
    """Return (versioned_rule_id, path_to_new_file).

    For modify_rule: adds v2, v3, ... inside the existing rule's folder.
    For new_rule: creates a new rule folder with v1.py.
    """
    if idea.kind == "modify_rule" and idea.target_rule:
        # target_rule is like "rule_01_spread_compression_v1" — strip version to get folder
        base = re.sub(r"_v\d+$", "", idea.target_rule)
        folder = _RULES_DIR / base
        existing = sorted(folder.glob("v*.py")) if folder.exists() else []
        if existing:
            m = re.match(r"v(\d+)\.py$", existing[-1].name)
            next_ver = (int(m.group(1)) + 1) if m else 2
        else:
            next_ver = 2
        rule_id = f"{base}_v{next_ver}"
        return rule_id, folder / f"v{next_ver}.py"
    else:
        existing_nums = [
            int(m.group(1))
            for d in _RULES_DIR.iterdir()
            if d.is_dir() and (m := re.match(r"rule_(\d+)_", d.name))
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        slug = re.sub(r"[^a-z0-9]+", "_", idea.title.lower()).strip("_")[:30]
        folder_name = f"rule_{next_num:02d}_{slug}"
        rule_id = f"{folder_name}_v1"
        return rule_id, _RULES_DIR / folder_name / "v1.py"


def _rule_id_to_import_path(rule_id: str) -> str:
    """Convert 'rule_01_spread_compression_v1' → 'rule_01_spread_compression.v1'."""
    m = re.match(r"^(.+)_(v\d+)$", rule_id)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return rule_id


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


def _apply_changes(code: str, changes: list[_CodeChange]) -> str:
    """Apply a list of line-level changes to code, returning the updated source."""
    lines = code.splitlines()
    # Process from bottom to top (highest line first) so earlier indices stay valid.
    # For equal line numbers, process removes before adds.
    for change in sorted(
        changes, key=lambda c: (c.line, c.action == "add"), reverse=True
    ):
        if change.action == "remove":
            idx = change.line - 1
            if 0 <= idx < len(lines):
                lines.pop(idx)
        elif change.action == "add":
            new_lines = change.code.splitlines() if change.code else []
            insert_at = (
                change.line
            )  # insert after line N → position N in 0-indexed list
            lines[insert_at:insert_at] = new_lines
    return "\n".join(lines)


def _load_target_source(target_rule: str) -> str | None:
    """Return the source of the rule being modified, or None if unavailable."""
    m = re.match(r"^(.+)_(v\d+)$", target_rule)
    if not m:
        return None
    path = _RULES_DIR / m.group(1) / f"{m.group(2)}.py"
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
    structured = llm.with_structured_output(_CodeDiff)
    user_prompt = (
        f"Idea:\n{idea.model_dump_json(indent=2)}\n\n"
        f"Available data models:\n{_MODELS_SOURCE}\n\n"
        f"Reference rule structure:\n{_REFERENCE_RULE}\n\n"
        "Entry-point function name: signal\n\n"
        f"Partial implementation to complete:\n{code}"
    )
    try:
        result = structured.invoke(
            [
                SystemMessage(content=_FIX_SYSTEM),
                HumanMessage(content=user_prompt),
            ]
        )
        if not isinstance(result, _CodeDiff):
            raise TypeError(f"Expected _CodeDiff, got {type(result).__name__}")
        logger.debug("Applying %d change(s) from diff", len(result.changes))
        return _apply_changes(code, result.changes)
    except Exception:
        logger.warning(
            "Diff generation/application failed; code unchanged", exc_info=True
        )
        return code


def _generate_code(idea: RuleIdea, rule_id: str, model: str) -> ImplementedRule:
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
            "Validation error (attempt %d/%d): %s",
            attempt + 1,
            _MAX_FIX_ATTEMPTS,
            error,
        )
        code = _fix_with_diff(code, idea, llm)
        logger.debug(
            "Code after fix attempt %d (%d chars):\n%s", attempt + 1, len(code), code
        )
    else:
        error = _check_syntax(code)
        if error is not None:
            raise _ImplementationFailed(
                f"Code still failing after {_MAX_FIX_ATTEMPTS} fix attempts: {error}"
            )

    return ImplementedRule(
        idea_id=idea.idea_id,
        rule_id=rule_id,
        function_name="signal",
        code=code,
    )


def _register_rule(strategy_path: Path, rule_id: str) -> None:
    """Add import and ACTIVE_RULES entry for the new rule version."""
    content = strategy_path.read_text(encoding="utf-8")
    import_path = _rule_id_to_import_path(rule_id)
    import_line = f"import src.strategy.rules.{import_path} as {rule_id}"

    if import_line not in content:
        last_import = re.search(
            r"^import src\.strategy\.rules\.\S+ as \S+$", content, re.MULTILINE
        )
        if last_import:
            content = (
                content[: last_import.end()]
                + "\n"
                + import_line
                + content[last_import.end() :]
            )
        else:
            content = import_line + "\n" + content

    if f"    {rule_id}," not in content:
        content = re.sub(
            r"(ACTIVE_RULES[^=]*=\s*\[)(.*?)(\n\])",
            rf"\1\2\n    {rule_id},\3",
            content,
            count=1,
            flags=re.DOTALL,
        )

    strategy_path.write_text(content, encoding="utf-8")
    logger.info("Registered %s in strategy.py", rule_id)


def _unregister_dropped(state_dir: Path, config: AppConfig) -> None:
    to_remove: list[str] = []

    rule_eval_path = state_dir / "rule_evaluation.json"
    if rule_eval_path.exists():
        try:
            evaluation = RuleEvaluation.model_validate_json(
                rule_eval_path.read_text(encoding="utf-8")
            )
            to_remove += [
                r.rule_id for r in evaluation.rules if r.status == "deprecate"
            ]
        except Exception:
            logger.warning("Could not read rule_evaluation.json for unregistration")

    version_cmp_path = state_dir / "version_comparison.json"
    if version_cmp_path.exists():
        try:
            cmp = VersionComparisonResult.model_validate_json(
                version_cmp_path.read_text(encoding="utf-8")
            )
            for c in cmp.comparisons:
                to_remove += c.versions_to_drop
        except Exception:
            logger.warning("Could not read version_comparison.json for unregistration")

    if not to_remove:
        return

    strategy_path = _STRATEGY_FILE
    if not strategy_path.exists():
        return

    for rule_id in set(to_remove):
        if f" as {rule_id}" not in strategy_path.read_text(encoding="utf-8"):
            logger.info("Rule %s not found in strategy.py; skipping", rule_id)
            continue
        _unregister_rule(strategy_path, rule_id)
        _remove_from_rule_evaluation(state_dir, rule_id)
        _remove_signals(rule_id, config)
        logger.info(
            "Unregistered %s from strategy.py, rule_evaluation.json, and signal ledger",
            rule_id,
        )


def _unregister_rule(strategy_path: Path, rule_id: str) -> None:
    lines = strategy_path.read_text(encoding="utf-8").splitlines(keepends=True)
    filtered = [
        line
        for line in lines
        if f" as {rule_id}" not in line
        and not re.fullmatch(rf"\s+{re.escape(rule_id)},?\n?", line)
    ]
    strategy_path.write_text("".join(filtered), encoding="utf-8")


def _remove_from_rule_evaluation(state_dir: Path, rule_id: str) -> None:
    rule_eval_path = state_dir / "rule_evaluation.json"
    if not rule_eval_path.exists():
        return
    try:
        evaluation = RuleEvaluation.model_validate_json(
            rule_eval_path.read_text(encoding="utf-8")
        )
        evaluation.rules = [r for r in evaluation.rules if r.rule_id != rule_id]
        rule_eval_path.write_text(
            evaluation.model_dump_json(indent=2), encoding="utf-8"
        )
    except Exception:
        logger.warning(
            "Could not update rule_evaluation.json after unregistering %s", rule_id
        )


def _remove_signals(rule_id: str, config: AppConfig) -> None:
    try:
        with open_db(config.data_dir) as con:
            con.execute("DELETE FROM signals WHERE rule_id = ?", (rule_id,))
        logger.info("Deleted signals for deprecated rule %s", rule_id)
    except Exception:
        logger.warning("Could not delete signals for rule %s", rule_id, exc_info=True)


def _commit_and_push(rule_id: str, function_name: str) -> None:
    commit_msg = f"agent: implement rule {rule_id} ({function_name})"
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "--set-upstream", "origin", "HEAD"], check=True)
        logger.info("Committed and pushed: %s", commit_msg)
    except subprocess.CalledProcessError:
        logger.exception("git commit/push failed after implementing %s", rule_id)
