"""Pydantic models for Strategy Updater pipeline structured LLM outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RuleDescription(BaseModel):
    rule_id: str
    description: str


class RuleDescriptions(BaseModel):
    rules: list[RuleDescription]


class Indicator(BaseModel):
    indicator_id: str
    name: str
    description: str
    code: str  # def compute(data: PairData) -> float | None


class IndicatorSet(BaseModel):
    version: str        # uuid4, changes when the set is modified
    indicators: list[Indicator]
    updated_at: str


class Plan(BaseModel):
    """plan.json — the single source of truth for both "which rule is
    active" and "what to do next cycle". The two used to be separate files
    (last_implemented.json, next_cycle_plan.json) but are always read
    together and rule_id/cycle_id only ever change alongside a fresh
    action/description, so one record covers both.
    """

    rule_id: str | None = None
    cycle_id: str | None = None
    action: Literal["continue", "fix", "new_rule"] = "new_rule"
    description: str | None = None


class RelationAnalysis(BaseModel):
    positive_patterns: list[str] = Field(
        description="Indicator combinations or conditions that correlated with positive outcomes"
    )
    negative_patterns: list[str] = Field(
        description="Indicator combinations or conditions that correlated with negative outcomes"
    )
    key_indicators: list[str] = Field(
        description="Most informative indicators for predicting outcome sign"
    )
    suggested_direction: str = Field(
        description="Proposed direction for the next rule idea, grounded in the observed patterns"
    )


class PendingRelationAnalysis(BaseModel):
    """relation_analysis.json — step4_relation_analysis.py's output, handed
    off to step5_generate_idea.py's step. cycle_id lets the reader tell a
    fresh analysis (produced this same pipeline run) apart from a stale
    leftover from a run where idea generation never got to consume it.
    """

    cycle_id: str
    plan: Plan
    analysis: RelationAnalysis


class TrainSample(BaseModel):
    signal_id: str
    cycle_id: str
    pair: str
    rule_id: str
    indicators: dict[str, float | None]
    opened_at: str   # signal's emitted_at
    closed_at: str   # outcome's evaluated_at (final settled outcome only)
    target_gain_pct: float


class TrainSet(BaseModel):
    samples: list[TrainSample] = []


class EpisodicTrace(BaseModel):
    trace_id: str
    cycle_id: str
    hypothesis: str
    rule_id: str
    indicator_set_version: str
    outcome_metrics: dict[str, float]
    diagnosis: str
    embedding: list[float] = []


class GainByVolatility(BaseModel):
    low: float | None = None
    medium: float | None = None
    high: float | None = None


class RuleScore(BaseModel):
    rule_id: str
    description: str
    signal_count: int
    emitted_signal_count: int = 0
    evaluation_days: int
    avg_gain_pct: float
    recent_avg_gain_pct: float
    avg_transaction_gain: float
    transaction_count: int = 0
    positive_rate: float
    avg_gain_24h: float
    max_gain_24h: float
    min_gain_pct: float = 0.0
    p25_gain_pct: float = 0.0
    p75_gain_pct: float = 0.0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    weekly_signal_counts: list[int] = []
    signal_trend: Literal["increasing", "decreasing", "stable"] = "stable"
    avg_gain_by_volatility: GainByVolatility = GainByVolatility()
    score: float


class RuleEvaluation(BaseModel):
    rules: list[RuleScore]
    summary: str


class VersionDirectionConclusion(BaseModel):
    rule_name: str
    dropped_versions: list[str]
    failed_direction: str
    proposed_direction: str


class Conclusions(BaseModel):
    conclusions: list[VersionDirectionConclusion]


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


class PendingRuleIdea(BaseModel):
    """rule_idea.json — the idea step5_generate_idea.py's step produced this
    cycle, handed off to step6_implement_rule.py's step. cycle_id lets the
    reader tell a fresh idea (generated this same pipeline run) apart from a
    stale leftover from a run where implementation never got to consume it.
    """

    cycle_id: str
    idea: RuleIdea
    plan: Plan
    analysis: RelationAnalysis


class ImplementedRule(BaseModel):
    idea_id: str
    rule_id: str          # module name, e.g. "rule_13_new_concept"
    function_name: str    # Python function name, e.g. "new_concept_signal"
    code: str             # complete Python file content
