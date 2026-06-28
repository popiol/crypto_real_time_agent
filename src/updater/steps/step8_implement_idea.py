"""Step 8 — Select and implement one idea.

Picks the highest-scored evaluated idea, generates the rule code via LLM,
writes the file, registers it in strategy.py, unregisters dropped/deprecated
versions, and marks the idea as implemented.

Writes:
  - a new rule file under src/strategy/rules/
  - updates src/strategy/strategy.py
  - updates idea_backlog.json
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from src.agent.models import AppConfig
from src.updater.llm import llm_structured
from src.updater.models import (
    IdeaBacklog,
    ImplementedRule,
    RuleEvaluation,
    RuleIdea,
    VersionComparisonResult,
)

logger = logging.getLogger(__name__)

_RULES_DIR = Path("src/strategy/rules")
_STRATEGY_FILE = Path("src/strategy/strategy.py")

_IMPLEMENT_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "Generate a complete, self-contained Python module that implements the described rule."
)

_REFERENCE_RULE = """\
\"\"\"Rule 01 — Spread compression spike.\"\"\"
from __future__ import annotations
import statistics
from src.agent.models import BuySignal, PairData

RULE_ID = "spread_compression_spike"
MIN_TICKS = 10
COMPRESSION_THRESHOLD = 0.30
MarketData = dict[str, PairData]

def spread_compression_spike(data: MarketData) -> list[BuySignal]:
    signals: list[BuySignal] = []
    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS:
            continue
        spreads = [t.spread_rel for t in ticks]
        baseline = statistics.mean(spreads[:-1])
        current = spreads[-1]
        if baseline > 0 and current < baseline * (1 - COMPRESSION_THRESHOLD):
            signals.append(BuySignal(
                pair=pair, rule_id=RULE_ID,
                timestamp=ticks[-1].polled_at, price=ticks[-1].last_price,
            ))
    return signals
"""


def run(config: AppConfig, state_dir: Path) -> None:
    backlog_path = state_dir / "idea_backlog.json"
    if not backlog_path.exists():
        logger.info("idea_backlog.json not found; skipping step 8")
        return

    backlog = IdeaBacklog.model_validate_json(backlog_path.read_text(encoding="utf-8"))

    # 1. Unregister dropped / deprecated versions
    _unregister_dropped(state_dir)

    # 2. Select best evaluated idea
    idea = _pick_best(backlog.ideas)
    if idea is None:
        logger.info("No evaluated ideas ready for implementation; skipping step 8")
        return

    # 3. Determine target file path
    rule_id, rule_path = _next_rule_path(idea)

    # 4. Generate code
    try:
        implemented = _generate_code(idea, rule_id, config.llm_model)
    except Exception:
        logger.error("Code generation failed for idea %s", idea.idea_id, exc_info=True)
        return

    # 5. Write rule file
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    rule_path.write_text(implemented.code, encoding="utf-8")
    logger.info("Wrote rule file: %s", rule_path)

    # 6. Register in strategy.py
    try:
        _register_rule(_STRATEGY_FILE, implemented.rule_id, implemented.function_name)
    except Exception:
        logger.error(
            "Failed to register %s in strategy.py", implemented.rule_id, exc_info=True
        )
        rule_path.unlink(missing_ok=True)
        return

    # 7. Mark idea as implemented
    for i in backlog.ideas:
        if i.idea_id == idea.idea_id:
            i.status = "implemented"
            break
    backlog_path.write_text(backlog.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "Implemented idea '%s' as %s.%s",
        idea.title,
        implemented.rule_id,
        implemented.function_name,
    )

    # 8. Commit and push all changes
    _commit_and_push(implemented.rule_id, implemented.function_name)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pick_best(ideas: list[RuleIdea]) -> RuleIdea | None:
    candidates = [i for i in ideas if i.status == "evaluated" and i.score is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda i: i.score or 0.0)


def _next_rule_path(idea: RuleIdea) -> tuple[str, Path]:
    if idea.kind == "modify_rule" and idea.target_rule:
        base = idea.target_rule
        existing = sorted(_RULES_DIR.glob(f"{base}_v*.py"))
        if existing:
            last_ver = int(re.search(r"_v(\d+)\.py$", existing[-1].name).group(1))
            next_ver = last_ver + 1
        else:
            # First version is the base file; next is v2
            next_ver = 2
        rule_id = f"{base}_v{next_ver}"
    else:
        existing_nums = [
            int(m.group(1))
            for f in _RULES_DIR.glob("rule_*.py")
            if (m := re.match(r"rule_(\d+)_", f.name))
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        slug = re.sub(r"[^a-z0-9]+", "_", idea.title.lower()).strip("_")[:30]
        rule_id = f"rule_{next_num:02d}_{slug}"

    return rule_id, _RULES_DIR / f"{rule_id}.py"


def _generate_code(idea: RuleIdea, rule_id: str, model: str) -> ImplementedRule:
    return llm_structured(
        model=model,
        system=_IMPLEMENT_SYSTEM,
        user=(
            f"Implement this trading rule idea as a Python module.\n\n"
            f"Idea:\n{idea.model_dump_json(indent=2)}\n\n"
            f"Module rule_id (use as filename stem): {rule_id}\n\n"
            f"Reference rule structure to follow exactly:\n{_REFERENCE_RULE}\n\n"
            "Requirements:\n"
            "1. Set RULE_ID to a unique snake_case string identifying this signal.\n"
            "2. Define MarketData = dict[str, PairData].\n"
            "3. Implement one public function with signature "
            "   def <func>(data: MarketData) -> list[BuySignal].\n"
            "4. Import only: standard library, numpy, scipy, statistics, "
            "   src.agent.models.\n"
            "5. Handle insufficient data gracefully (return []).\n"
            f"6. Set rule_id field to '{rule_id}', function_name to the function's name, "
            "   idea_id to the idea's idea_id."
        ),
        output_type=ImplementedRule,
    )


def _register_rule(strategy_path: Path, rule_id: str, function_name: str) -> None:
    content = strategy_path.read_text(encoding="utf-8")

    import_line = f"from src.strategy.rules.{rule_id} import {function_name}"
    last_import = re.search(
        r"^from src\.strategy\.rules\.\S+ import \S+$", content, re.MULTILINE
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

    active_close = content.rfind("]", content.find("ACTIVE_RULES ="))
    content = (
        content[:active_close] + f"    {function_name},\n" + content[active_close:]
    )

    strategy_path.write_text(content, encoding="utf-8")
    logger.info("Registered %s in strategy.py", function_name)


def _unregister_dropped(state_dir: Path) -> None:
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

    for module_rule_id in set(to_remove):
        fn_name = _get_function_name(strategy_path, module_rule_id)
        if fn_name is None:
            logger.info("Rule %s not found in strategy.py; skipping", module_rule_id)
            continue
        _unregister_rule(strategy_path, fn_name)
        logger.info("Unregistered %s (%s) from strategy.py", module_rule_id, fn_name)


def _get_function_name(strategy_path: Path, module_rule_id: str) -> str | None:
    content = strategy_path.read_text(encoding="utf-8")
    m = re.search(
        rf"from src\.strategy\.rules\.{re.escape(module_rule_id)} import (\w+)", content
    )
    return m.group(1) if m else None


def _unregister_rule(strategy_path: Path, function_name: str) -> None:
    lines = strategy_path.read_text(encoding="utf-8").splitlines(keepends=True)
    filtered = [
        line
        for line in lines
        if f"import {function_name}" not in line
        and not re.fullmatch(rf"\s+{re.escape(function_name)},?\n?", line)
    ]
    strategy_path.write_text("".join(filtered), encoding="utf-8")


def _commit_and_push(rule_id: str, function_name: str) -> None:
    commit_msg = f"agent: implement rule {rule_id} ({function_name})"
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push"], check=True)
        logger.info("Committed and pushed: %s", commit_msg)
    except subprocess.CalledProcessError:
        logger.exception("git commit/push failed after implementing %s", rule_id)
