---
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - docs/design.md
---

# crypto_real_time_agent - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for crypto_real_time_agent, decomposing the requirements from the design document into implementable stories.

## Requirements Inventory

### Functional Requirements (changes only)

FR1: The pipeline must maintain a persisted indicator set (indicator_set.json) — an LLM-defined list of indicators, each with a name, description, and a generated Python function `compute(data: PairData) -> float | None`.
FR2: At the start of each failure cycle, the LLM must review the indicator set and may add, remove, or modify indicators; on success cycles (avg_gain_pct > 0.5%) the set is left unchanged.
FR3: Each cycle, all indicator functions must be executed against the hot/warm/cold tier snapshot from 24h ago, producing a flat dict of indicator values per pair.
FR4: Each cycle must append one train sample per signal to train_set.json: indicator values from 24h ago for the signalling pair and the last 24h price change for that pair.
FR5: The LLM must perform relation analysis using the computed indicators, the train set, and top-K semantically retrieved episodic traces to identify which patterns preceded positive/negative outcomes.
FR6: Episodic traces must be retrieved via cosine similarity over hypothesis embeddings; top-K (default 5) most similar traces are included.
FR7: After each cycle, an immutable episodic trace must be written to data/state/traces/<cycle_id>.json: hypothesis, rule_id, indicator_set_version, outcome_metrics, diagnosis, and hypothesis embedding.
FR8: Each cycle, the LLM must generate exactly one rule idea (new_rule or fix) derived from the relation analysis output, and implement it immediately in the same pipeline run.
FR9: On failure, the LLM must first attempt to diagnose whether the failure is fixable; if fixable, produce a fix (new version); if not, produce a new_rule idea.
FR10: A next_cycle_plan.json must be written at the end of each cycle with action (continue / fix / new_rule) and a description; this is the primary input to the following cycle's step 1.
FR11: rule_evaluation.json must be extended with: min_gain_pct, p25_gain_pct, p75_gain_pct, longest_win_streak, longest_loss_streak, weekly_signal_counts, signal_trend, avg_gain_by_volatility (low/medium/high BBW buckets).
FR12: The following files must be removed from the pipeline and no longer produced: version_comparison.json, conclusions.json, long_term_plan.json, idea_backlog.json.

### NonFunctional Requirements (changes only)

NFR1: LLM-generated indicator code runs in-process; execution errors must be caught per-indicator and must not halt the pipeline.
NFR2: Embeddings must be computed for each episodic trace hypothesis and stored alongside the trace for efficient cosine similarity retrieval in future cycles.

### Additional Requirements

- An embedding model must be integrated for episodic trace semantic search (local or API-based — to be decided).
- LLM-generated indicator code signature: `def compute(data: PairData) -> float | None`.
- The pipeline moves from 8 steps to 9 steps; the scheduler trigger comment in Section 11 of design.md must be updated to reflect this.
- top-K trace retrieval count (default 5) and success threshold (0.5%) must be configurable via config.yaml.

### UX Design Requirements

N/A — headless autonomous agent.

### FR Coverage Map

```
FR1:  Epic 2 — Indicator set definition and persistence
FR2:  Epic 2 — Conditional indicator set modification on failure
FR3:  Epic 2 — Indicator code execution against tier snapshot
FR4:  Epic 3 — Train sample appended per signal
FR5:  Epic 4 — Relation analysis on train set + retrieved traces
FR6:  Epic 3 — Trace retrieval via cosine similarity (embeddings)
FR7:  Epic 3 — Immutable episodic trace writing with embedding
FR8:  Epic 4 — One idea generated and implemented per cycle
FR9:  Epic 4 — Fix-or-new-rule decision on failure
FR10: Epic 4 — next_cycle_plan.json written each cycle
FR11: Epic 1 — Enhanced rule evaluation metrics
FR12: Epic 1 — Remove version_comparison, conclusions, long_term_plan, idea_backlog
```

## Epic List

### Epic 1: Richer Rule Evaluation
The system produces more informative rule performance metrics, enabling better diagnosis in later epics.
**FRs covered:** FR11, FR12

### Epic 2: LLM-Controlled Indicator System
The system lets the LLM define what to measure — generating, persisting, and executing its own indicator code each cycle.
**FRs covered:** FR1, FR2, FR3

### Epic 3: Learning Memory
The system accumulates a train set of signal contexts and outcomes, and writes immutable episodic traces with embeddings for future retrieval.
**FRs covered:** FR4, FR6, FR7

### Epic 4: Data-Driven Rule Generation Loop
The system closes the learning loop — relation analysis drives rule ideas, one idea is generated and implemented per cycle, and a plan is written to steer the next cycle.
**FRs covered:** FR5, FR8, FR9, FR10

---

## Epic 1: Richer Rule Evaluation

The system produces more informative rule performance metrics, enabling better diagnosis in later epics.

### Story 1.1: Add distribution and streak metrics to rule evaluation

As an autonomous agent operator,
I want rule evaluation to include p25, p75, min gain, and longest win/loss streak per rule,
So that the system can distinguish between rules that are consistently mediocre and rules that are volatile or streaky.

**Acceptance Criteria:**

**Given** the signal ledger contains evaluated signals for a rule
**When** the rule evaluation step runs
**Then** `rule_evaluation.json` includes `min_gain_pct`, `p25_gain_pct`, `p75_gain_pct` computed across all evaluated signals for that rule

**Given** a rule has a sequence of evaluated signals ordered by `emitted_at`
**When** the rule evaluation step runs
**Then** `longest_win_streak` is the maximum number of consecutive signals with `gain_24h_pct > 0`, and `longest_loss_streak` is the maximum number of consecutive signals with `gain_24h_pct <= 0`

**Given** a rule has fewer than 2 evaluated signals
**When** the rule evaluation step runs
**Then** distribution and streak metrics are set to `0`

**Given** the existing `RuleScore` Pydantic model
**When** the model is updated
**Then** all new fields are added with correct types and the model remains backwards-compatible with existing pipeline code that reads `rule_evaluation.json`

---

### Story 1.2: Add signal frequency and volatility-bucketed gain metrics

As an autonomous agent operator,
I want rule evaluation to include weekly signal frequency trends and average gain split by market volatility,
So that the system can detect whether a rule is firing less over time and whether it only works in specific market conditions.

**Acceptance Criteria:**

**Given** the signal ledger contains evaluated signals for a rule
**When** the rule evaluation step runs
**Then** `weekly_signal_counts` is a list of integers (one per calendar week, oldest first) counting signals emitted in each week since the rule was first active

**Given** `weekly_signal_counts` has at least 3 entries
**When** the rule evaluation step runs
**Then** `signal_trend` is `increasing` if the last 3 weeks average exceeds the first 3 weeks average, `decreasing` if it is lower, and `stable` otherwise; if fewer than 3 entries exist, `signal_trend` is `stable`

**Given** the warm tier data is available for each signal's pair at the time the signal was emitted
**When** the rule evaluation step runs
**Then** `avg_gain_by_volatility` contains three fields — `low`, `medium`, `high` — each being the average `gain_24h_pct` for signals emitted when the Bollinger Band Width at signal time falls in the bottom, middle, and top third of the BBW distribution for that rule's signal history respectively; a bucket with no signals has value `null`

**Given** the `RuleScore` Pydantic model
**When** the model is updated
**Then** `weekly_signal_counts: list[int]`, `signal_trend: Literal["increasing", "decreasing", "stable"]`, and `avg_gain_by_volatility: GainByVolatility` are added, and `GainByVolatility` is a Pydantic model with fields `low: float | None`, `medium: float | None`, `high: float | None`

---

### Story 1.3: Remove deprecated pipeline steps and output files

As an autonomous agent operator,
I want the deprecated pipeline steps and their output files removed,
So that the pipeline no longer produces stale artifacts that mislead diagnosis.

**Acceptance Criteria:**

**Given** the strategy updater pipeline
**When** it runs
**Then** the following steps are no longer executed: compare rule versions, derive conclusions, update long-term plan, evaluate ideas, select idea from backlog

**Given** the state directory
**When** the pipeline runs after this story is implemented
**Then** `version_comparison.json`, `conclusions.json`, `long_term_plan.json`, and `idea_backlog.json` are not written; any existing files with those names are left in place but ignored

**Given** all pipeline code that reads those files
**When** those reads are removed
**Then** no remaining pipeline step references those files and no import or function call to the removed step modules exists

---

## Epic 2: LLM-Controlled Indicator System

The system lets the LLM define what to measure — generating, persisting, and executing its own indicator code each cycle.

### Story 2.1: Bootstrap and persist the initial indicator set

As an autonomous agent operator,
I want the pipeline to initialise a persisted indicator set on first run,
So that subsequent cycles have a defined set of indicators to compute and analyse.

**Acceptance Criteria:**

**Given** `data/state/indicator_set.json` does not exist
**When** the pipeline runs for the first time
**Then** the LLM is prompted to define an initial set of indicators relevant to mean-reversion trading on the hot/warm/cold tier data, and the result is written to `data/state/indicator_set.json`

**Given** `data/state/indicator_set.json` is written
**When** the file is read
**Then** it deserialises into `IndicatorSet` with a `version` UUID, `updated_at` timestamp, and a non-empty list of `Indicator` entries each containing `indicator_id`, `name`, `description`, and a `code` string containing a valid Python function with signature `def compute(data: PairData) -> float | None`

**Given** `data/state/indicator_set.json` already exists
**When** the pipeline runs
**Then** the bootstrap step is skipped entirely

---

### Story 2.2: Update indicator set on failure cycles

As an autonomous agent operator,
I want the LLM to review and optionally revise the indicator set at the start of each failure cycle,
So that the features available for relation analysis can evolve when the current set is not sufficient.

**Acceptance Criteria:**

**Given** `next_cycle_plan.json` has `action: fix` or `action: new_rule`
**When** the pipeline runs step 1
**Then** the LLM receives the current `indicator_set.json` and the plan description, and may add, remove, or modify indicators; the result is written back to `indicator_set.json` with a new `version` UUID and updated `updated_at`

**Given** `next_cycle_plan.json` has `action: continue`
**When** the pipeline runs step 1
**Then** `indicator_set.json` is not modified and step 1 completes immediately

**Given** the LLM produces an updated indicator set
**When** the updated set is written
**Then** each indicator's `code` field is validated for Python syntax; any entry failing the check is rejected and the previous version of that indicator is retained, with a warning logged

---

### Story 2.3: Execute indicator functions against tier snapshot

As an autonomous agent operator,
I want each indicator function to be executed against the tier snapshot from 24h ago for every pair that emitted a signal,
So that computed indicator values are available for relation analysis and train set accumulation.

**Acceptance Criteria:**

**Given** `indicator_set.json` is loaded and signals were emitted in the previous cycle
**When** step 2 runs
**Then** each indicator's `compute(data: PairData) -> float | None` function is called with the tier snapshot from 24h ago for each signalling pair, producing a dict `{indicator_name: value}` per pair

**Given** an indicator function raises an exception or returns a non-float non-None value
**When** step 2 runs
**Then** that indicator's value for that pair is set to `None`, the error is logged, and execution continues with remaining indicators and pairs

**Given** step 2 completes
**When** the in-memory results are passed to step 3
**Then** the result is a dict keyed by pair, each value being a flat `dict[str, float | None]` covering all indicators in the current set

---

## Epic 3: Learning Memory

The system accumulates a train set of signal contexts and outcomes, and writes immutable episodic traces with embeddings for future retrieval.

### Story 3.1: Accumulate train samples per signal

As an autonomous agent operator,
I want each evaluated signal to contribute a train sample to an accumulating dataset,
So that the relation analysis has a growing body of evidence linking indicator patterns to outcomes.

**Acceptance Criteria:**

**Given** a signal has been evaluated (outcome filled in the signal ledger) and indicator values are available for the signalling pair from that cycle
**When** step 7 (update train set) runs
**Then** one `TrainSample` is appended to `data/state/train_set.json` containing: `cycle_id`, `pair`, `indicators` (flat dict from step 2), and `target_gain_pct` (the signal's `gain_24h_pct`)

**Given** `data/state/train_set.json` does not exist
**When** the first sample is appended
**Then** the file is created with a `TrainSet` structure containing the single sample

**Given** multiple signals were emitted in the same cycle for different pairs
**When** step 7 runs
**Then** one sample is appended per signal

---

### Story 3.2: Compute and store hypothesis embeddings

As an autonomous agent operator,
I want each episodic trace to include a text embedding of its hypothesis,
So that semantically similar past traces can be retrieved efficiently in future cycles.

**Acceptance Criteria:**

**Given** a rule idea (hypothesis text) has been generated in step 4
**When** step 8 begins writing the episodic trace
**Then** the hypothesis text is embedded using the configured embedding model, producing a vector of floats stored in the `embedding` field of the trace

**Given** the embedding model is configured via `config.yaml`
**When** the embedding is computed
**Then** the same model is used consistently across all cycles so that cosine similarity comparisons are valid

**Given** the embedding computation fails
**When** step 8 runs
**Then** the error is logged, `embedding` is set to `[]`, and trace writing continues

---

### Story 3.3: Write immutable episodic traces

As an autonomous agent operator,
I want an immutable trace record written after each cycle,
So that the system has a permanent, retrievable record of every hypothesis tested and its outcome.

**Acceptance Criteria:**

**Given** a cycle has completed (rule evaluated, train set updated, embedding computed)
**When** step 8 runs
**Then** a file is written to `data/state/traces/<cycle_id>.json` containing: `trace_id` (UUID), `cycle_id`, `hypothesis`, `rule_id`, `indicator_set_version`, `outcome_metrics`, `diagnosis`, and `embedding`

**Given** `data/state/traces/<cycle_id>.json` already exists
**When** step 8 runs
**Then** the existing file is never overwritten; an error is logged and the step is skipped for that cycle

**Given** the traces directory does not exist
**When** the first trace is written
**Then** `data/state/traces/` is created automatically

---

## Epic 4: Data-Driven Rule Generation Loop

The system closes the learning loop — relation analysis drives rule ideas, one idea is generated and implemented per cycle, and a plan is written to steer the next cycle.

### Story 4.1: Retrieve relevant traces and run relation analysis

As an autonomous agent operator,
I want the LLM to analyse indicator-to-outcome patterns from the train set, informed by the most relevant past traces,
So that rule ideas are grounded in observed data rather than generated without context.

**Acceptance Criteria:**

**Given** `data/state/train_set.json` and `data/state/traces/` contain data from previous cycles
**When** step 3 runs
**Then** the top-K traces (default 5, configurable via `config.yaml`) are retrieved by computing cosine similarity between the current plan's text embedding and each trace's stored `embedding`; traces with `embedding: []` are skipped

**Given** the top-K traces and the full train set are assembled
**When** the LLM performs relation analysis
**Then** it receives the computed indicator values for all signalling pairs, the train set, and the top-K trace texts, and returns a structured analysis identifying which indicator patterns correlated with positive and negative outcomes

**Given** `data/state/train_set.json` is empty or does not exist
**When** step 3 runs
**Then** relation analysis proceeds with indicator values alone and no train set context; no error is raised

---

### Story 4.2: Generate one rule idea and implement it immediately

As an autonomous agent operator,
I want exactly one rule idea generated per cycle from the relation analysis and implemented in the same pipeline run,
So that each cycle produces a concrete, testable hypothesis without queuing ideas.

**Acceptance Criteria:**

**Given** the relation analysis output from step 3
**When** step 4 runs
**Then** the LLM generates exactly one `RuleIdea` — either `kind: new_rule` or `kind: fix` — with `title`, `description`, `rationale`, `pseudocode`, and `target_rule` (null for new_rule, rule name for fix)

**Given** the generated idea
**When** step 5 runs
**Then** the LLM generates a complete, executable Python rule file and registers it in `strategy.py`; for `new_rule`, a new folder and `v1.py` are created; for `fix`, a new version file is added alongside the previous version

**Given** step 5 completes
**When** `strategy.py` is updated
**Then** the new rule version appears in `ACTIVE_RULES` and its import is present; idempotency guards prevent duplicate registrations if the step is re-run

---

### Story 4.3: Diagnose failure and plan fix-or-new-rule

As an autonomous agent operator,
I want the system to attempt to diagnose and fix a failed rule before discarding its direction,
So that fixable implementation issues are corrected rather than abandoned prematurely.

**Acceptance Criteria:**

**Given** the implemented rule's `avg_gain_pct` is ≤ 0.5% after evaluation
**When** step 9 runs
**Then** the LLM receives the rule's outcome metrics, the episodic trace diagnosis, and the relation analysis, and determines whether the failure is fixable or fundamental

**Given** the LLM determines the failure is fixable
**When** step 9 writes `next_cycle_plan.json`
**Then** `action` is `fix` and `description` contains a specific, actionable description of what to change in the next version

**Given** the LLM determines the failure is not fixable
**When** step 9 writes `next_cycle_plan.json`
**Then** `action` is `new_rule` and `description` contains the direction for the next relation analysis to explore

---

### Story 4.4: Write next_cycle_plan on success

As an autonomous agent operator,
I want a next_cycle_plan written even on successful cycles,
So that the pipeline always has a valid plan to read at the start of the next run.

**Acceptance Criteria:**

**Given** the implemented rule's `avg_gain_pct` > 0.5%
**When** step 9 runs
**Then** `next_cycle_plan.json` is written with `action: continue` and `description: null`

**Given** `next_cycle_plan.json` is written with `action: continue`
**When** the next cycle's step 1 reads it
**Then** step 1 is skipped and the pipeline proceeds directly to step 2

**Given** `next_cycle_plan.json` does not exist
**When** step 1 runs
**Then** it treats the missing file as `action: new_rule`, triggering indicator set initialisation as per Story 2.1
