"""Pydantic models for Strategy Updater pipeline structured LLM outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class RuleSignalEvaluation(BaseModel):
    rule_id: str
    signal_count: int
    notes: str  # LLM qualitative interpretation of this rule's signal outcomes


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


class Conclusions(BaseModel):
    what_works: str
    what_doesnt: str
    open_questions: str


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
    kind: Literal["new_rule", "modify_rule"]
    target_rule: str | None
    score: float | None
    status: Literal["proposed", "evaluated", "implemented", "rejected"]


class IdeaBacklog(BaseModel):
    ideas: list[RuleIdea]


class ImplementedRule(BaseModel):
    idea_id: str
    rule_id: str          # module name, e.g. "rule_13_new_concept"
    function_name: str    # Python function name, e.g. "new_concept_signal"
    code: str             # complete Python file content
