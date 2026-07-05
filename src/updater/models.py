"""Pydantic models for Strategy Updater pipeline structured LLM outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PairMetrics(BaseModel):
    pair: str
    signal_count: int
    avg_gain_pct: float
    positive_rate: float


class RuleSignalEvaluation(BaseModel):
    rule_id: str
    signal_count: int
    positive_rate: float
    avg_gain_pct: float
    by_exit_reason: dict[str, int]
    by_pair: list[PairMetrics]


class SignalEvaluation(BaseModel):
    rules: list[RuleSignalEvaluation]


class RuleDescription(BaseModel):
    rule_id: str
    description: str


class RuleDescriptions(BaseModel):
    rules: list[RuleDescription]


class RuleScore(BaseModel):
    rule_id: str
    description: str
    signal_count: int
    avg_gain_pct: float
    positive_rate: float
    avg_gain_24h: float
    max_gain_24h: float
    score: float
    status: Literal["candidate", "active", "deprecate"]


class RuleEvaluation(BaseModel):
    rules: list[RuleScore]
    summary: str


class Conclusion(BaseModel):
    text: str
    rule_ids: list[str]


class Conclusions(BaseModel):
    conclusions: list[Conclusion]


class LongTermPlan(BaseModel):
    direction: str
    priorities: list[str]
    updated_at: str


class RuleVersionComparison(BaseModel):
    rule_name: str
    versions_compared: list[str]
    best_version: str
    versions_to_drop: list[str]
    rationale: str


class VersionComparisonResult(BaseModel):
    comparisons: list[RuleVersionComparison]
    summary: str


class RuleIdea(BaseModel):
    idea_id: str
    title: str
    description: str
    rationale: str
    pseudocode: str
    kind: Literal["new_rule", "modify_rule"] = "new_rule"
    target_rule: str | None = None
    score: float | None = None
    status: Literal["proposed", "evaluated", "implemented", "rejected"] = "proposed"


class IdeaBacklog(BaseModel):
    ideas: list[RuleIdea]


class ImplementedRule(BaseModel):
    idea_id: str
    rule_id: str          # module name, e.g. "rule_13_new_concept"
    function_name: str    # Python function name, e.g. "new_concept_signal"
    code: str             # complete Python file content
