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

import logging
import re
import subprocess
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

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

_RULES_DIR = Path("src/strategy/rules")
_STRATEGY_FILE = Path("src/strategy/strategy.py")

_IMPLEMENT_SYSTEM = (
    "You are an expert Python developer specialising in quantitative trading rules. "
    "Generate a complete, self-contained Python module that implements the described rule."
)

_REFERENCE_RULE = """\
\"\"\"Rule 01 — Spread compression spike (v1).\"\"\"
from __future__ import annotations
import statistics
from src.agent.models import BuySignal, SellSignal, PairData

RULE_ID = "rule_01_spread_compression_v1"
MIN_TICKS = 10
COMPRESSION_THRESHOLD = 0.30
MarketData = dict[str, PairData]

def spread_compression_spike(data: MarketData) -> list[BuySignal | SellSignal]:
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

    # 3. Determine target file path and versioned rule_id
    rule_id, rule_path = _next_rule_path(idea)

    # 4. Generate code
    try:
        implemented = _generate_code(idea, rule_id, config.llm_model)
    except Exception:
        logger.exception("Code generation failed for idea %s", idea.idea_id)
        return

    # 5. Write rule file
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    rule_path.write_text(implemented.code, encoding="utf-8")
    logger.info("Wrote rule file: %s", rule_path)

    # 6. Register in strategy.py
    try:
        _register_rule(_STRATEGY_FILE, rule_id, implemented.function_name)
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
        idea.title, rule_id, implemented.function_name,
    )

    # 8. Commit and push all changes
    _commit_and_push(rule_id, implemented.function_name)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pick_best(ideas: list[RuleIdea]) -> RuleIdea | None:
    candidates = [i for i in ideas if i.status == "evaluated" and i.score is not None]
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


def _generate_code(idea: RuleIdea, rule_id: str, model: str) -> ImplementedRule:
    """Generate rule code as plain text to avoid JSON string-length limits in structured output."""
    llm = make_llm(model)
    user_prompt = (
        f"Implement this trading rule idea as a Python module.\n\n"
        f"Idea:\n{idea.model_dump_json(indent=2)}\n\n"
        f"Rule ID: {rule_id}\n\n"
        f"Reference rule structure to follow exactly:\n{_REFERENCE_RULE}\n\n"
        "Requirements:\n"
        f"1. Set RULE_ID = \"{rule_id}\" exactly.\n"
        "2. Define MarketData = dict[str, PairData].\n"
        "3. Implement one public function with signature "
        "   def <func>(data: MarketData) -> list[BuySignal | SellSignal].\n"
        "4. Import only: standard library, numpy, scipy, statistics, "
        "   src.agent.models.\n"
        "5. Handle insufficient data gracefully (return []).\n"
        "6. Return ONLY the raw Python source code — no markdown fences, no explanation."
    )
    response = llm.invoke([SystemMessage(content=_IMPLEMENT_SYSTEM), HumanMessage(content=user_prompt)])
    raw = response.content
    code = (raw if isinstance(raw, str) else "".join(p if isinstance(p, str) else p.get("text", "") for p in raw)).strip()

    # Strip accidental markdown fences
    if code.startswith("```"):
        code = re.sub(r"^```[a-z]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code)

    # Extract function name from the generated code
    m = re.search(r"^def (\w+)\(data:", code, re.MULTILINE)
    if not m:
        raise ValueError(f"Could not find public function in generated code for {rule_id}")
    function_name = m.group(1)

    return ImplementedRule(
        idea_id=idea.idea_id,
        rule_id=rule_id,
        function_name=function_name,
        code=code,
    )


def _register_rule(strategy_path: Path, rule_id: str, function_name: str) -> None:
    """Add import and ACTIVE_RULES entry for the new rule version."""
    content = strategy_path.read_text(encoding="utf-8")
    import_path = _rule_id_to_import_path(rule_id)
    import_line = f"from src.strategy.rules.{import_path} import {function_name}"

    last_import = re.search(
        r"^from src\.strategy\.rules\.\S+ import \S+$", content, re.MULTILINE
    )
    if last_import:
        content = (
            content[: last_import.end()] + "\n" + import_line + content[last_import.end():]
        )
    else:
        content = import_line + "\n" + content

    content = re.sub(
        r"(ACTIVE_RULES[^=]*=\s*\[)(.*?)(\n\])",
        rf"\1\2\n    {function_name},\3",
        content,
        count=1,
        flags=re.DOTALL,
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
            to_remove += [r.rule_id for r in evaluation.rules if r.status == "deprecate"]
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
        fn_name = _get_function_name(strategy_path, rule_id)
        if fn_name is None:
            logger.info("Rule %s not found in strategy.py; skipping", rule_id)
            continue
        _unregister_rule(strategy_path, fn_name)
        logger.info("Unregistered %s (%s) from strategy.py", rule_id, fn_name)


def _get_function_name(strategy_path: Path, rule_id: str) -> str | None:
    """Find the function name imported for the given versioned rule_id."""
    import_path = re.escape(_rule_id_to_import_path(rule_id))
    content = strategy_path.read_text(encoding="utf-8")
    m = re.search(rf"from src\.strategy\.rules\.{import_path} import (\w+)", content)
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
        subprocess.run(["git", "push", "--set-upstream", "origin", "HEAD"], check=True)
        logger.info("Committed and pushed: %s", commit_msg)
    except subprocess.CalledProcessError:
        logger.exception("git commit/push failed after implementing %s", rule_id)
