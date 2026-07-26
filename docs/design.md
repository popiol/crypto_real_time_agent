# Design Document — Crypto Real-Time Agent

## 1. Overview

The Crypto Real-Time Agent is a locally-run Python application that continuously monitors selected cryptocurrency pairs on Kraken, applies a dynamic rule-based strategy to detect buy opportunities, and automatically evaluates the quality of each rule based on observed market outcomes. The strategy is not fixed — it is a living artifact that grows smarter as more signal-outcome data accumulates.

---

## 2. Components

```
┌─────────────────────────────────────────────────────┐
│                    Scheduler / Main Loop             │
│  (polls Kraken, triggers analysis, triggers review)  │
└────────────┬────────────────────┬───────────────────┘
             │                    │
             ▼                    ▼
  ┌──────────────────┐   ┌─────────────────────┐
  │  Data Collector  │   │   Strategy Engine   │
  │  (Kraken API)    │   │  (strategy.py)      │
  └────────┬─────────┘   └──────────┬──────────┘
           │                        │
           ▼                        ▼
  ┌──────────────────┐   ┌─────────────────────┐
  │  Tiered Storage  │   │   Signal Ledger     │
  │  (SQLite)        │   │   (SQLite)          │
  └──────────────────┘   └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Signal Evaluator   │
                         │  (outcome tracker)  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Strategy Updater   │
                         │  (LLM pipeline)     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   strategy.py       │
                         │  (active rules)     │
                         └─────────────────────┘
```

---

## 3. Data Collection

### 3.1 Source

Kraken REST API — public endpoints (no authentication required for market data):
- `GET /public/Ticker` — returns bid (`b`), ask (`a`), last trade price (`c`), VWAP (`p`), volume (`v`), high/low (`h`/`l`), open (`o`), and trade count (`t`) for each requested pair.
- `GET /public/OHLC` — returns up to 720 OHLC candles per request (1-minute default, configurable interval); used to bootstrap warm/cold storage on first run.

### 3.2 Polling rate

Kraken's public API allows up to 1 request per second per endpoint per IP (confirmed in Kraken support documentation). Rather than polling on a fixed timer, the collector triggers a new poll immediately after the previous analysis cycle finishes. This avoids queuing up polls when analysis takes longer than the poll interval, and naturally adapts to however fast the system can process data. Backoff on `EGeneral:Too many requests` responses is still applied using exponential backoff.

### 3.3 Tracked data per poll

- Timestamp (UTC)
- Last trade price
- Best bid price and volume
- Best ask price and volume
- 24-hour rolling volume (base currency, from Kraken Ticker)
- Order book snapshot (top levels, bid/ask)
- Derived: mid-price, bid/ask spread (absolute and relative)

---

## 4. Tiered Storage

All data is stored locally. The guiding principle is: the older the data, the more it is compressed into statistics.

### 4.1 Tiers

| Tier | Coverage | Granularity | Contents |
|---|---|---|---|
| **Hot** | Most recent N ticks (configurable, default last ~5 minutes) | Every poll | Full tick: timestamp, bid, ask, last price, spread |
| **Warm** | Last 24 hours | 1 entry per hour | Hourly OHLC of last price, average spread |
| **Cold** | Older than 24 hours | 1 entry per calendar month | Min price, max price, avg price, avg daily spread, candle count, last candle hour — aggregated, not a time series |

### 4.2 Downsampling

A background job runs every hour to:
1. Aggregate hot-tier ticks older than 24 hours into the warm tier (capped at the last 24 hourly candles).
2. Recompute cold-tier monthly aggregates from any warm-tier candles that have rolled off, merging into the existing per-month row.

### 4.3 Storage backend

All three tiers live as tables in a single SQLite database, `data/assets/agent.db` (`src/agent/db.py`), not separate per-pair files:

- **Hot tier** — `hot_ticks` table, one row per poll per pair, indexed on `(pair, polled_at)`.
- **Warm tier** — `warm_candles` table, one row per pair per hour (`PRIMARY KEY (pair, hour)`); reads are capped to the most recent 24 rows per pair (`ORDER BY hour DESC LIMIT 24`).
- **Cold tier** — `cold_months` table, one row per pair per calendar month (`PRIMARY KEY (pair, month)`).
- The signal ledger's `signals` table (§6) lives in the same database.

The connection runs in WAL mode (`PRAGMA journal_mode=WAL`) for concurrent reader/writer access between the polling loop and the hourly downsampling job.

---

## 5. Strategy Engine

### 5.1 Interface

The strategy is a single Python module `strategy/strategy.py` that exposes one function:

```python
def find_signals(data: MarketData, config: AppConfig) -> list[BuySignal | SellSignal]:
    ...
```

`MarketData` is a typed object containing all three storage tiers for all tracked pairs. Each signal carries:
- `pair` — the currency pair
- `rule_id` — the identifier of the rule that fired
- `timestamp` — when the signal was generated
- `price` — price at signal time
- `confidence` — optional float 0–1 (rule may omit this)

### 5.2 Rules

Each rule version is a self-contained Python file under `strategy/rules/<rule_name>/`. Each file exposes a single function:

```python
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    ...
```

Exactly one rule is active at a time. The active rule is *state*, not code: `find_signals()` reads `data/state/plan.json`'s `rule_id` (§8.1) on every call and dynamically imports whichever rule module it names, then calls that module's `signal()` function. The Strategy Updater never edits `strategy.py` itself — this matches the Strategy Updater's one-hypothesis-per-cycle learning loop (§8): there is always exactly one hypothesis under test, never a portfolio of concurrently running rules. Rules are purely functional — they read data and return signals; they have no side effects.

**Data tier limits.** Every rule receives `data.hot` (≤~300 recent ticks, ~5 minutes), `data.warm` (at most 24 hourly OHLC candles — never more), and `data.cold` (one aggregate row per calendar month: min/max/avg price, avg daily spread, candle count — not a time series, §4.1). An indicator whose lookback exceeds 24 hourly candles cannot be computed from `data.warm`, and `data.cold` cannot substitute for it since it holds no individual hourly/daily closes — only coarse monthly aggregates. A rule that ignores this will pass syntax/type validation but silently return `[]` forever against real data (§8.2 Step 6 prompts the generating LLM with this constraint explicitly, including during fix attempts).

Directory layout:

```
strategy/
  rules/
    rule_01_spread_compression/
      v1.py
      v2.py
    rule_02_momentum_reversal/
      v1.py
  strategy.py
```

### 5.3 Rule versioning

Each rule version has a unique `rule_id` formed from the rule name and version (e.g., `rule_01_spread_compression_v2`). When the Strategy Updater implements a new rule idea (§8.2 Step 6), it writes the new `rule_id` to `data/state/plan.json` — a `fix` idea's version file is added alongside the previous version of the same rule, a `new_rule` idea's file goes in a new folder. Either way, the new version immediately becomes the sole active rule, since `find_signals()` re-reads `plan.json` on every call. The previous version's file is kept on disk under `strategy/rules/` for signal traceability but is no longer imported or executed once replaced.

---

## 6. Signal Ledger

Every buy signal emitted by the strategy is written to a persistent ledger — the `signals` table in a SQLite database at `data/assets/agent.db` (see `src/agent/db.py`).

### 6.1 Signal record (at emission time)

```json
{
  "signal_id": "uuid4",
  "pair": "XBTUSD",
  "rule_id": "spread_compression_v1",
  "emitted_at": "2026-06-15T10:32:00Z",
  "price_at_signal": 67420.50,
  "confidence": 0.72,
  "indicators": { "rsi_14": 38.2, "sma_short": 67100.0 },
  "outcome": null
}
```

`indicators` is a snapshot of every current indicator's value against the pair's tier data *at the moment the signal was created* (`src/agent/loop.py`, right after `find_signals()` returns) — not recomputed later. A signal takes up to ~24h to resolve (§7), by which point `data.warm`'s 24-hour rolling window has already moved partway past the market state that actually produced it, so this is the only point where a causally-correct reading is available.

### 6.2 Outcome record

The two outcome fields resolve independently, on very different timelines, and `outcome` is exposed by `storage.read_signals()` as soon as *either* is present:

```json
{
  "outcome": {
    "gain_24h_pct": 2.49,
    "max_gain_24h_pct": 4.20,
    "evaluated_at": "2026-06-16T10:32:00Z",
    "exit_price": 69100.00,
    "exit_reason": "sell_signal",
    "gain_pct": 2.61
  }
}
```

- `gain_24h_pct` / `max_gain_24h_pct` — a fixed 24h-later price read, resolved once a signal is 24-48h old regardless of whether a real exit has happened. Available for the vast majority of evaluated signals within roughly a day.
- `evaluated_at` / `exit_price` / `exit_reason` / `gain_pct` — the final settled outcome: either a matching opposite-direction signal for the same pair, or a 24h timeout (§7). Only present once one of those actually happens.

A signal can have `gain_24h_pct` set and `gain_pct` absent for a long time — that's the normal case, not a partial/broken record. Code that needs "is there anything to judge yet" should check `outcome is not None`; code that specifically needs the final settled result should check `outcome.get("gain_pct") is not None`.

`gain_24h_pct` = `(price_24h - price_at_signal) / price_at_signal * 100`
`max_gain_24h_pct` = `(max_price_24h - price_at_signal) / price_at_signal * 100`
`gain_pct` = `(exit_price - price_at_signal) / price_at_signal * 100`

---

## 7. Signal Evaluator

A background job runs every hour and resolves two independent outcome fields per pending buy signal:

1. **24h read** (`gain_24h_pct`, `max_gain_24h_pct`): once a signal is 24-48h old, reconstructs prices from the warm tier over that window and writes both fields. This does not require a sell signal or any exit to have happened — it's a fixed snapshot.
2. **Final outcome** (`gain_pct`, `exit_price`, `exit_reason`, `evaluated_at`): resolved either by a matching opposite-direction signal for the same pair emitted after it (`exit_reason: "sell_signal"`), or — once nothing has matched by 24h — by a timeout using the latest warm-tier close (`exit_reason: "timeout"`). Since this timeout is deliberately as short as the 24h read (not the 20-day ceiling used earlier), almost every signal reaches a final settled outcome within about a day, rather than staying in 24h-snapshot-only limbo for the rest of its (usually much shorter) lifetime — §8.2 Step 1 prefers `gain_pct` once it exists and only falls back to `gain_24h_pct` in the brief window before it does.

---

## 8. Strategy Updater

A periodic LLM-driven pipeline that evaluates strategy performance and evolves `strategy.py` autonomously. It runs as a sequence of steps, each persisting its output to a local file so the pipeline is resumable and auditable.

### 8.1 Persistence artifacts

| File / Directory | Contents |
|---|---|
| `data/state/rule_evaluation.json` | Accumulated per-rule-version scoring, description, and enhanced metrics — one entry per rule ever evaluated; descriptions are cached here across runs |
| `data/state/indicator_set.json` | The current set of LLM-defined indicators: name, description, and generated Python code for each |
| `data/state/train_set.json` | Accumulating samples of (indicator values captured at signal emission, open/close timestamps, final settled gain) — one per finally-settled signal, across all cycles |
| `data/state/traces/` | Episodic trace store — one JSON file per cycle, immutable; contains hypothesis, indicator set version, outcome metrics, and LLM diagnosis |
| `data/state/plan.json` | `{rule_id, cycle_id, action, description}` — `plan_next_cycle.py`'s single record of both "which rule is active" and "what to do next cycle". `rule_id`/`cycle_id` identify the most recently implemented rule (signal outcomes lag implementation by design, 24h+ to resolve, so this is how later steps know which rule they're following up on); `action`/`description` are `continue`, or `fix`/`new_rule` with a description of what to attempt |
| `data/state/relation_analysis.json` | `{cycle_id, plan, analysis}` — step 3's output, handed off to step 5; cleared once consumed |
| `data/state/rule_idea.json` | `{cycle_id, idea, plan, analysis}` — step 5's output, handed off to step 6; cleared once consumed |

### 8.2 Pipeline steps

All steps that involve reasoning use the LLM (see section 9). Each step reads its inputs from persisted files and writes its output before the pipeline moves on, making the whole run resumable and auditable. The pipeline runs every 24 hours.

The 6 steps below are numbered by `pipeline.py`'s actual call order — every physical filename carries the same number (`stepN_...py`), so there is exactly one numbering scheme, visible from the filename alone. `plan_next_cycle.py` is not one of the 6: its decision logic runs at multiple points across them (a re-check at the start of step 3, and outcome-recording in steps 5 and 6), so it doesn't fit a single numbered slot — see the unnumbered subsection at the end of this list. There is no dedicated "write episodic trace" step either — that used to be its own step, but it only ever ran on the first cycle after a rule was implemented, when no signal could possibly have resolved yet (resolution takes ~24h), so every trace ever written showed zero signals. Trace-writing now happens inside "Plan next cycle," at the moment a rule is actually retired, capturing its real final performance instead.

Indicator set revision (step 4) deliberately runs *after* the continue-recheck (step 3), not before: it used to run first (position 1), which meant it only ever saw `plan.json`'s action as it stood at the *start* of the cycle — before step 3's own recheck could flip it. A rule replaced later in that same cycle would never get its indicator set reconsidered, since the newly-implemented rule's `plan.json` gets reset straight back to `continue`, and the next cycle's now-earlier indicator step would just see `continue` again. Running it after step 3 means it always sees this cycle's *final* verdict.

#### Step 1 — Evaluate outcomes and score the active rule
*Inputs*: signal ledger (signals with a resolved outcome, §6.2), prior `rule_evaluation.json`  
*Output*: `rule_evaluation.json` (updated — one entry, for the current active rule)

Computes metrics for the currently active rule:
- `signal_count` — signals with *any* outcome (`gain_24h_pct` or the final `gain_pct`, §6.2); `emitted_signal_count` — all signals emitted regardless of resolution. Since the 24h read resolves within about a day while the final outcome can take up to 20, `signal_count` tracks the 24h read in practice — the "plan next cycle" logic below relies on the gap between it and `emitted_signal_count` to tell "hasn't resolved yet" apart from "never fires."
- `evaluation_days`
- `avg_gain_pct`, `recent_avg_gain_pct` (last 48h of data) — per signal, the final `gain_pct` if it has resolved, else `gain_24h_pct`
- `min_gain_pct` (worst result), `p25_gain_pct`, `p75_gain_pct`
- `positive_rate`, `avg_gain_24h`, `max_gain_24h`
- `longest_win_streak`, `longest_loss_streak`
- `weekly_signal_counts` — list of signal counts per week, oldest first
- `signal_trend` — `increasing`, `decreasing`, or `stable` (derived from weekly counts)
- `avg_gain_by_volatility` — avg gain split into low/medium/high Bollinger Band Width buckets at signal time
- `score` — composite score normalised to [0, 1]
- `description` — a plain-language description of the rule, generated once per version and cached in `rule_evaluation.json` across runs

`rule_evaluation.json` accumulates one entry per rule ever evaluated: the active rule's entry is refreshed each cycle; every other rule_id's entry is carried over unchanged from whenever it was last active. There is no `status` field — whether to keep or replace the active rule is decided purely from `recent_avg_gain_pct` (see "Plan next cycle" below), not from a separate classification.

#### Step 2 — Update train set
*Inputs*: signal ledger (signals newly resolved this cycle, with their `indicators` captured at emission time, §6.1)  
*Output*: `data/state/train_set.json` (new samples appended), `data/state/indicator_set.json` (dead indicators pruned)

For each signal that now has a *final settled* outcome (`gain_pct` — a real sell-signal match or the 24h timeout, not just the fixed `gain_24h_pct` snapshot) and isn't already in the train set, appends `{signal_id, cycle_id, pair, rule_id, indicators, opened_at, closed_at, target_gain_pct}` using the `indicators` already stored on the signal record — nothing is recomputed here. `opened_at`/`closed_at` are the signal's `emitted_at` and the outcome's `evaluated_at`. The train set accumulates indefinitely across cycles. Any indicator that comes back null across every sample added this cycle is pruned from `indicator_set.json`, since that's the only point a fresh batch of real indicator readings is available to check.

#### Step 3 — Relation analysis
*Inputs*: `data/state/plan.json`, `data/state/train_set.json`, top-K episodic traces retrieved from `data/state/traces/` by semantic similarity to the current plan  
*Output*: `data/state/plan.json` (re-checked), `data/state/relation_analysis.json` — `{cycle_id, plan, analysis}`, possibly a new file in `data/state/traces/<cycle_id>.json`

First re-checks whether the active rule is still performing (see "Plan next cycle" below — this is also where a retired rule's episodic trace gets written) — if it's still performing, the step stops here and nothing downstream (steps 4-6) runs this cycle. Otherwise, the LLM receives a sample of the accumulated train set (indicator values paired with their resolved outcome) and the most relevant past traces, and identifies which indicator patterns preceded positive and negative outcomes, and what those relations suggest about the next rule to try. Indicator values only ever reach this step through `train_set.json` — they're captured once, at signal emission time (§6.1), and carried through to whichever cycle later resolves that signal's outcome (step 2), since only samples with a known outcome are useful for correlation. Traces are retrieved by embedding the current plan and performing cosine similarity search over trace hypothesis embeddings; top-K (default 5) most similar traces are included.

`relation_analysis.json` is stamped with the `cycle_id` it was produced in, so step 5 can tell a fresh analysis apart from a stale leftover from a run where nothing consumed it.

#### Step 4 — Update indicator set
*Inputs*: `data/state/plan.json` (this cycle's *final* action, after step 3's recheck), `data/state/indicator_set.json`  
*Output*: `data/state/indicator_set.json` (updated)

Skipped if `plan.json` has `action: continue` (this cycle's rule is still performing — gain > 0.5%). Otherwise, the LLM reviews the current indicator set in light of the plan and may add, remove, or modify indicators. For each change, the LLM generates a Python function that computes the indicator from `PairData` (hot/warm/cold tiers). The updated set is persisted.

#### Step 5 — Generate rule idea
*Inputs*: `data/state/relation_analysis.json` (skipped if missing or stamped with a different cycle_id than the current one)  
*Output*: `data/state/rule_idea.json` — `{cycle_id, idea, analysis}`, `data/state/relation_analysis.json` (cleared)

The LLM generates exactly one idea per cycle, derived from the relation analysis. Two kinds:
- **New rule** — an entirely new rule concept; if implemented, a new rule folder and `v1.py` are created. `target_rule` is null.
- **Fix** — a targeted change to the currently active rule; if implemented, a new version file is added alongside the existing one. `target_rule` is always the current active rule's `rule_id`, set directly from `plan.json` rather than asked of the LLM — since only one rule is ever active, there is never any ambiguity about what a "fix" targets.

`rule_idea.json` is stamped with the `cycle_id` it was generated in, so step 6 can tell a fresh idea apart from a stale leftover from a run where nothing consumed it.

#### Step 6 — Implement rule
*Inputs*: `data/state/rule_idea.json` (skipped if missing or stamped with a different cycle_id than the current one), source of the existing rule (for fix ideas)  
*Output*: new rule version file under `strategy/rules/`, `data/state/plan.json` (updated), `data/state/rule_idea.json` (cleared)

Generates real, executable Python code from the idea persisted in step 5. Exactly one rule version is added per pipeline run, and it becomes the sole active rule by being written to `plan.json`, replacing whichever rule was previously active. The previous version's file is kept under `strategy/rules/` for signal traceability but is no longer imported or executed. Also records the cycle's outcome via "Plan next cycle" below — either `action: continue` for the newly-implemented rule, or (if implementation failed) a fresh diagnosis for the still-active rule.

#### Plan next cycle
*(not one of the 6 numbered steps — `plan_next_cycle.py` is called from step 3, step 5's failure path, and step 6, not once in sequence)*

Decision logic, evaluated fresh every cycle against whichever rule is currently active:
- If `rule_evaluation.json` has no entry for the rule yet, or the rule has zero *evaluated* signals (`signal_count == 0`) but is actively emitting them (`emitted_signal_count > 0`): `action: continue` — nothing to judge yet, since a signal can only resolve via a matching opposite-direction signal or a 24h timeout (§7), and treating an unresolved rule as 0% gain would replace every rule before it ever gets a fair look.
- If the rule genuinely never emits any signal at all (`signal_count == 0` and `emitted_signal_count == 0`): falls through to the check below like any other rule — this is a real failure (e.g. an indicator window exceeding the 24-candle warm-tier cap, §5.2), not a timing artifact, and should be diagnosed and replaced.
- Otherwise, the metric judged is `recent_avg_gain_pct` (last 48h of signal-theoretical gains, §8.2 Step 1) alone — not blended with `avg_transaction_gain` (the portfolio's own trading gate, §10.3, still uses the combined formula; this decision no longer does).
  - If `recent_avg_gain_pct` > 0.5%: `action: continue` — leave the indicator set and the active rule alone.
  - Otherwise: one LLM call produces both a narrative diagnosis (why the rule performed as it did) and the fix/new_rule verdict — whether the failure is fixable (wrong thresholds, wrong indicators) or the hypothesis itself was wrong.
  - If fixable: `action: fix` with a description of the specific change to attempt.
  - If not fixable: `action: new_rule` — relation analysis starts fresh from the data.

When a rule is retired this way (action moves from `continue` to `fix`/`new_rule`), that same LLM call's narrative diagnosis, together with the rule's final `rule_evaluation.json` snapshot, is written as an immutable episodic trace to `data/state/traces/<cycle_id>.json` — tagged with the *current* cycle_id (the retirement moment), not the rule's original implementation cycle_id:
```json
{
  "trace_id": "uuid4",
  "cycle_id": "2026-07-23T10:00:00Z",
  "hypothesis": "...",
  "rule_id": "rule_47_..._v1",
  "indicator_set_version": "uuid4",
  "outcome_metrics": { "avg_gain_pct": ..., "positive_rate": ..., "p25": ..., "p75": ... },
  "diagnosis": "...",
  "embedding": [...]
}
```
This is the only point traces are ever written — a rule that's still active never has one, and once written, a trace is never edited. The `embedding` field is computed from the rule's description for semantic retrieval by step 3 in future cycles.

Step 3 re-runs this check immediately at the start of every cycle rather than trusting the previous cycle's stale verdict: if the rule is still performing, the cycle ends there; if it's no longer performing, the cycle proceeds straight into relation analysis → idea → implementation in the same run, rather than waiting a full cycle to act. Whenever a rule is actually replaced (step 6), the plan written for the *next* cycle is always `continue` for the new rule — the old rule's diagnosis is never carried over and misapplied to its replacement, since the new rule hasn't had any chance yet to earn a verdict of its own.

### 8.3 Rule lifecycle

```
[idea generated] → [implemented] → [continue | replaced]
```

Ideas are generated in step 5 and implemented immediately in step 6 — they are not queued or persisted separately; there is no backlog. Once implemented, a rule stays active for as long as "Plan next cycle" keeps deciding `continue`. There is no separate status classification or grace-period counter: replacement is decided fresh every cycle from `recent_avg_gain_pct` alone (§8.2 "Plan next cycle"). A replaced rule's file remains under `strategy/rules/` for signal traceability, but is no longer imported or executed once `plan.json` (§5.3) points elsewhere.

---

## 9. LLM Interface

All model calls across the Strategy Updater pipeline are made through **LangChain**, which abstracts over providers and handles prompt templating. The underlying model is swappable via configuration without changing application code.

### 9.1 Model

The model name is read from `config.yaml` and passed to the LangChain chat model constructor at startup.

### 9.2 Structured output

All LLM responses that feed into application logic are parsed into **Pydantic models**. Each pipeline step has its own output model:

```python
class Indicator(BaseModel):
    indicator_id: str
    name: str
    description: str
    code: str           # Python function: def compute(data: PairData) -> float | None

class IndicatorSet(BaseModel):
    version: str        # uuid4, changes when the set is modified
    indicators: list[Indicator]
    updated_at: str

class Plan(BaseModel):
    # plan.json — both "which rule is active" and "what to do next cycle"
    rule_id: str | None          # None until step 6 first implements a rule
    cycle_id: str | None         # cycle_id the active rule was implemented in
    action: Literal["continue", "fix", "new_rule"]
    description: str | None     # what to attempt; None when action is "continue"

class RelationAnalysis(BaseModel):
    positive_patterns: list[str]   # indicator combinations that correlated with positive outcomes
    negative_patterns: list[str]   # indicator combinations that correlated with negative outcomes
    key_indicators: list[str]      # most informative indicators for predicting outcome sign
    suggested_direction: str       # proposed direction for the next rule idea

class PendingRelationAnalysis(BaseModel):
    # relation_analysis.json — step 3's output, handed off to step 5
    cycle_id: str
    plan: Plan
    analysis: RelationAnalysis

class TrainSample(BaseModel):
    signal_id: str
    cycle_id: str
    pair: str
    rule_id: str
    indicators: dict[str, float | None]
    opened_at: str      # signal's emitted_at
    closed_at: str      # outcome's evaluated_at (final settled outcome only)
    target_gain_pct: float

class TrainSet(BaseModel):
    samples: list[TrainSample]

class GainByVolatility(BaseModel):
    low: float | None
    medium: float | None
    high: float | None

class RuleScore(BaseModel):
    rule_id: str                        # includes version, e.g. "rule_01_spread_compression_v2"
    description: str                    # cached from prior run or generated fresh
    signal_count: int                   # signals with any outcome (24h read or final), §6.2
    emitted_signal_count: int           # all emitted signals, regardless of resolution
    evaluation_days: int
    avg_gain_pct: float                 # theoretical: per signal, final gain_pct if resolved else gain_24h_pct
    recent_avg_gain_pct: float          # avg_gain_pct over signals in the last 48h of data
    avg_transaction_gain: float         # realized, from the portfolio's closed transactions (net of fees)
    transaction_count: int              # 0 means avg_transaction_gain is not yet meaningful
    min_gain_pct: float                 # worst single signal outcome
    p25_gain_pct: float
    p75_gain_pct: float
    positive_rate: float
    avg_gain_24h: float
    max_gain_24h: float
    longest_win_streak: int
    longest_loss_streak: int
    weekly_signal_counts: list[int]     # one entry per week, oldest first
    signal_trend: Literal["increasing", "decreasing", "stable"]
    avg_gain_by_volatility: GainByVolatility
    score: float                        # composite, normalised to [0, 1]

class RuleEvaluation(BaseModel):
    rules: list[RuleScore]
    summary: str

class EpisodicTrace(BaseModel):
    trace_id: str
    cycle_id: str
    hypothesis: str
    rule_id: str
    indicator_set_version: str
    outcome_metrics: dict[str, float]
    diagnosis: str
    embedding: list[float]

class RuleIdea(BaseModel):
    idea_id: str
    title: str
    description: str
    rationale: str
    pseudocode: str
    kind: Literal["new_rule", "modify_rule"]
    target_rule: str | None     # rule_id of the rule to fix; None for new_rule
    score: float | None
    status: Literal["proposed", "evaluated", "implemented", "rejected"]

class PendingRuleIdea(BaseModel):
    # rule_idea.json — step 5's output, handed off to step 6
    cycle_id: str
    idea: RuleIdea
    plan: Plan
    analysis: RelationAnalysis

class ImplementedRule(BaseModel):
    idea_id: str
    rule_id: str                # module name, e.g. "rule_13_new_concept"
    function_name: str          # Python function name
    code: str                   # complete Python file content
```

---

## 10. Virtual Portfolio

A simulated trading portfolio that runs on every data pull cycle (not just analysis cycles). It tracks cash, open positions, and pending limit orders, and persists state to `data/portfolio/`.

### 10.1 Persistence

| File | Contents |
|---|---|
| `data/portfolio/portfolio.json` | Current portfolio state: cash, total value, open positions, pending orders |
| `data/portfolio/transactions.json` | Append-only log of closed positions (filled sell orders) |

### 10.2 Data models

```python
class Position(BaseModel):
    position_id: str
    pair: str
    rule_id: str
    quantity: float
    buy_price: float
    value: float        # quantity * current_price, updated each cycle
    opened_at: datetime

class Order(BaseModel):
    order_id: str
    direction: Literal["buy", "sell"]
    pair: str
    rule_id: str
    limit_price: float
    quantity: float
    value: float        # quantity * limit_price
    position_id: str | None  # set for sell orders
    created_at: datetime

class Portfolio(BaseModel):
    cash: float
    value: float        # cash + sum of open position values at current prices
    positions: list[Position]
    pending_orders: list[Order]

class Transaction(BaseModel):
    transaction_id: str
    pair: str
    rule_id: str
    quantity: float
    buy_price: float
    sell_price: float
    cost: float         # quantity * buy_price * (1 + fee)
    revenue: float       # net after fee
    gain_pct: float
    opened_at: datetime
    closed_at: datetime
```

### 10.3 Cycle logic

Each data pull cycle executes in order:

1. **Fill orders** — for each pending order, check the current tick price:
   - Buy fills when `current_price < limit_price` (strict). Cost = `quantity × limit_price × (1 + fee)`.
   - Sell fills when `current_price > limit_price` (strict). Revenue = `quantity × limit_price × (1 - fee)`.
   - A filled sell creates a `Transaction` record appended to `transactions.json`.
   - If cash would go negative, the buy fill is skipped with a warning.

2. **Auto-close stale positions** — any position held for more than 24 hours (relative to the latest tick timestamp, not wall clock) gets a sell order placed at the current price. Any existing sell order for that position is cancelled first.

3. **Place new orders** — looks up the active rule (`plan.json`'s `rule_id`) in `rule_evaluation.json`. There is exactly one active rule at a time (§5.3), so this never picks among candidates — it only checks whether that one rule's `recent_avg_gain_pct + avg_transaction_gain` exceeds `portfolio_min_recent_gain`. If it does, signals from that rule in the current cycle are acted on:
   - Buy signal → place a buy limit order at the signal price, spending `capital / 10`. Maximum 10 simultaneous open positions. Cash already committed to pending buy orders is deducted before checking available cash.
   - Sell signal → place a sell limit order at the signal price. If a sell order already exists for that position, it is cancelled and replaced.

4. **Update values** — all position `value` fields and `portfolio.value` are recomputed from current tick prices, then state is saved.

### 10.4 Configuration

| Field | Default | Meaning |
|---|---|---|
| `portfolio_initial_capital` | `10000.0` | Starting cash in USD |
| `portfolio_min_recent_gain` | `0.005` | Minimum `recent_avg_gain_pct + avg_transaction_gain` for the active rule to trigger orders |
| `portfolio_fee` | `0.0025` | Exchange fee applied to both sides of each trade (0.25%) |

---

## 11. Scheduler / Main Loop



The main process runs a cooperative loop with the following periodic tasks:

| Task | Trigger |
|---|---|
| Poll Kraken API + run strategy | Immediately after the previous cycle completes (rate-limit aware) |
| Downsample hot → warm tier | Every hour |
| Recompute cold-tier statistics | Every hour (after warm downsampling) |
| Evaluate pending signal outcomes | Every hour |
| Run Strategy Updater pipeline (§8.2, steps 1-6) | Every 24 hours |

---

## 12. Configuration

A single `config.yaml` at the project root controls:

- Poll interval and backoff parameters
- Hot-tier max tick retention count
- Cycle success threshold (`cycle_success_threshold`) for the continue/fix/new_rule decision (§8.2 "Plan next cycle")
- Data directory path (SQLite database + portfolio files live here)
- State directory path (default: `data/state/`)
- LLM model name (e.g.: `gemini-2.5-flash`)
- Backtesting data directory (default: `../crypto_alerts_llm/data/raw`)
- Portfolio settings: `portfolio_initial_capital`, `portfolio_min_recent_gain`, `portfolio_fee`

---

## 13. Open Questions / Future Work

- **Signal deduplication**: if the same rule fires on consecutive ticks, subsequent signals from that rule for the same pair are suppressed for 24 hours.
- **Alerting**: emit a notification (e.g., desktop notification, webhook) when a high-confidence buy signal fires.
- **Backtesting**: replay historical data through the strategy to evaluate rules without waiting for real-time outcomes. The backtesting module reads from an external data directory (configurable, default `../crypto_alerts_llm/data/raw`). That directory contains hourly snapshots partitioned as `year=YYYY/month=MM/day=DD/<timestamp>.json` (full Kraken Ticker response for all pairs) and `<timestamp>_bidask.json` (top-5 order book levels per pair). The backtester feeds these snapshots through the tiered storage layer and strategy engine in chronological order, producing a signal ledger that can be evaluated immediately since all future prices are available.
- **Multiple timeframes**: currently warm tier is hourly; finer granularity (e.g., 5-minute) could improve some rule types.
