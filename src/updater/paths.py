"""Centralised state-file paths for the Strategy Updater pipeline.

Every step reads and/or writes JSON state under `state_dir`. Filenames are
defined once here so steps and the pipeline orchestrator (which archives
them) can't drift out of sync.
"""

from __future__ import annotations

from pathlib import Path

INDICATOR_SET = "indicator_set.json"
PLAN = "plan.json"
RULE_EVALUATION = "rule_evaluation.json"
RULE_DESCRIPTIONS = "rule_descriptions.json"
VERSION_COMPARISON = "version_comparison.json"
TRAIN_SET = "train_set.json"
CONCLUSIONS = "conclusions.json"
RELATION_ANALYSIS = "relation_analysis.json"
RULE_IDEA = "rule_idea.json"
LONG_TERM_PLAN = "long_term_plan.json"


def indicator_set(state_dir: Path) -> Path:
    return state_dir / INDICATOR_SET


def plan(state_dir: Path) -> Path:
    return state_dir / PLAN


def rule_evaluation(state_dir: Path) -> Path:
    return state_dir / RULE_EVALUATION


def version_comparison(state_dir: Path) -> Path:
    return state_dir / VERSION_COMPARISON


def train_set(state_dir: Path) -> Path:
    return state_dir / TRAIN_SET


def conclusions(state_dir: Path) -> Path:
    return state_dir / CONCLUSIONS


def relation_analysis(state_dir: Path) -> Path:
    return state_dir / RELATION_ANALYSIS


def rule_idea(state_dir: Path) -> Path:
    return state_dir / RULE_IDEA


def long_term_plan(state_dir: Path) -> Path:
    return state_dir / LONG_TERM_PLAN


def traces_dir(state_dir: Path) -> Path:
    return state_dir / "traces"


def trace_file(state_dir: Path, cycle_id: str) -> Path:
    return traces_dir(state_dir) / f"{cycle_id}.json"


def history_dir(state_dir: Path, ts: str) -> Path:
    return state_dir / "history" / ts
