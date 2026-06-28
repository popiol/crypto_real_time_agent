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
  │  (local files)   │   │  (signals.json/db)  │
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
- Derived: mid-price, bid/ask spread (absolute and relative)

---

## 4. Tiered Storage

All data is stored locally. The guiding principle is: the older the data, the more it is compressed into statistics.

### 4.1 Tiers

| Tier | Coverage | Granularity | Contents |
|---|---|---|---|
| **Hot** | Most recent N ticks (configurable, default last ~5 minutes) | Every poll | Full tick: timestamp, bid, ask, last price, spread |
| **Warm** | Last 24 hours | 1 entry per hour | Hourly OHLC of last price, average spread |
| **Cold** | Older than 24 hours | 1 entry per statistical window | Min price, max price, avg price, avg daily spread — kept for: last 7 days, last 30 days, last 90 days, last 365 days, all-time |

### 4.2 Downsampling

A background job runs every hour to:
1. Aggregate hot-tier ticks older than 24 hours into the warm tier.
2. Recompute all cold-tier statistical windows (7d, 30d, 90d, 365d, all-time) from the warm tier and any previously stored cold-tier data.

### 4.3 File format

- **Hot tier**: append-only newline-delimited JSON (`.ndjson`) per currency pair.
- **Warm tier**: one JSON array file per pair, replaced on each downsampling pass.
- **Cold tier**: one JSON array file per pair, one entry per calendar month.

Directory layout:

```
data/
  <PAIR>/
    hot.ndjson
    warm.json
    cold.json
```

---

## 5. Strategy Engine

### 5.1 Interface

The strategy is a single Python module `strategy/strategy.py` that exposes one function:

```python
def find_buy_signals(data: MarketData) -> list[BuySignal]:
    ...
```

`MarketData` is a typed object containing all three storage tiers for all tracked pairs. `BuySignal` carries:
- `pair` — the currency pair
- `rule_id` — the identifier of the rule that fired
- `timestamp` — when the signal was generated
- `price` — price at signal time
- `confidence` — optional float 0–1 (rule may omit this)

### 5.2 Rules

Each rule version is a self-contained Python file under `strategy/rules/<rule_name>/`. Each file exposes a single function:

```python
def rule(data: MarketData) -> list[BuySignal]:
    ...
```

`strategy.py` imports the registered version files and calls their `rule` functions, merging all outputs. Rules are purely functional — they read data and return signals; they have no side effects.

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

Each active rule version has a unique `rule_id` formed from the rule name and version (e.g., `rule_01_spread_compression_v2`). When a rule is revised, a new version file (`v2.py`, `v3.py`, …) is added to the rule's folder and registered in `strategy.py`. The old version file is kept for signal traceability but is unregistered once it is dropped or superseded. Multiple versions of the same rule can be active simultaneously while their relative performance is being assessed.

---

## 6. Signal Ledger

Every buy signal emitted by the strategy is written to a persistent ledger (`data/signals.json` or a SQLite database — TBD based on scale).

### 6.1 Signal record (at emission time)

```json
{
  "signal_id": "uuid4",
  "pair": "XBTUSD",
  "rule_id": "spread_compression_v1",
  "emitted_at": "2026-06-15T10:32:00Z",
  "price_at_signal": 67420.50,
  "confidence": 0.72,
  "outcome": null
}
```

### 6.2 Outcome record (filled in 24 hours later)

```json
{
  "outcome": {
    "evaluated_at": "2026-06-16T10:32:00Z",
    "price_24h": 69100.00,
    "max_price_24h": 70250.00,
    "gain_24h_pct": 2.49,
    "max_gain_24h_pct": 4.20
  }
}
```

`gain_24h_pct` = `(price_24h - price_at_signal) / price_at_signal * 100`
`max_gain_24h_pct` = `(max_price_24h - price_at_signal) / price_at_signal * 100`

---

## 7. Signal Evaluator

A background job runs every hour and:
1. Queries the ledger for signals emitted more than 24 hours ago with `outcome = null`.
2. Retrieves the warm/cold tier data for the relevant pair to reconstruct prices in the 24-hour window after the signal.
3. Computes `gain_24h_pct` and `max_gain_24h_pct` and writes the outcome back to the ledger.

---

## 8. Strategy Updater

A periodic LLM-driven pipeline that evaluates strategy performance and evolves `strategy.py` autonomously. It runs as a sequence of steps, each persisting its output to a local file so the pipeline is resumable and auditable.

### 8.1 Persistence artifacts

| File | Contents |
|---|---|
| `data/state/signal_evaluation.json` | Aggregated interpretation of recent signal outcomes |
| `data/state/rule_descriptions.json` | Cached plain-language descriptions of each rule version; generated once per version, reused across runs |
| `data/state/rule_evaluation.json` | Per-rule-version scoring, status, and description based on signal history |
| `data/state/version_comparison.json` | Results of comparing versions of the same rule; marks inferior versions for dropping |
| `data/state/conclusions.json` | Strategic conclusions derived from the latest rule evaluation |
| `data/state/long_term_plan.json` | Evolving high-level direction for the strategy |
| `data/state/idea_backlog.json` | Pool of rule ideas with status (`proposed`, `evaluated`, `implemented`, `rejected`) |

### 8.2 Pipeline steps

All steps that involve reasoning use the LLM (see section 9). Steps run sequentially; each reads its inputs from persisted files and writes its output before the next step begins.

#### Step 1 — Analyze results
*Inputs*: signal ledger (signals with filled-in outcomes)  
*Output*: `signal_evaluation.json`

Aggregates the signal outcome data into a structured summary: overall win rate, average and max gains by pair, by time of day, by market conditions at signal time. Identifies patterns in which signals performed well or poorly.

#### Step 2 — Analyze current rules
*Inputs*: `signal_evaluation.json`, `rule_descriptions.json`, registered rule versions from `strategy/rules/`  
*Output*: `rule_descriptions.json` (updated), `rule_evaluation.json`

Processes each registered rule version independently in two sub-steps:

1. **Describe** — for each version not already present in `rule_descriptions.json`, makes one LLM call to produce a concise plain-language description of what the rule does and the signal it looks for. Newly generated descriptions are written to `rule_descriptions.json` immediately; existing descriptions are reused as-is.

2. **Score** — for each version: computes signal count, average gain, max gain, positive rate, and a composite score from the signal ledger data in `signal_evaluation.json`. Determines whether the version has enough data to be scored. Flags versions that fall below the deprecation threshold.

The final `rule_evaluation.json` contains both the description and the scoring for every version.

#### Step 3 — Compare rule versions
*Inputs*: `rule_evaluation.json`, registered rule versions from `strategy/rules/`  
*Output*: `version_comparison.json`

For each rule that has more than one registered version, compares composite scores from `rule_evaluation.json`. If a version's score is significantly worse than the best-performing version of the same rule, it is marked for dropping. Versions still in candidate status (insufficient signal data) are excluded from comparison — they are not dropped until they have been scored. The output also captures the rationale for each drop decision. Versions marked for dropping are unregistered in step 8.

#### Step 4 — Derive conclusions
*Inputs*: `rule_evaluation.json`, `version_comparison.json`, registered rule versions from `strategy/rules/`  
*Output*: `conclusions.json`

Interprets the rule evaluation holistically: what kinds of signals tend to work, what market conditions correlate with good outcomes, what structural weaknesses exist in the current rule set.

#### Step 5 — Update long term plan
*Inputs*: `conclusions.json`, existing `long_term_plan.json`  
*Output*: `long_term_plan.json` (updated)

Revises the strategic direction for the agent based on the latest conclusions. The plan is a persistent document that accumulates knowledge across many update cycles.

#### Step 6 — Generate ideas
*Inputs*: `long_term_plan.json`, `idea_backlog.json`, registered rule versions from `strategy/rules/`  
*Output*: `idea_backlog.json` (new entries appended)

There are two kinds of ideas:
- **New rule** — proposes an entirely new rule concept. If implemented, a new rule folder and `v1.py` are created.
- **Modify rule** — proposes a revision to an existing rule. If implemented, a new version file (`v2.py`, `v3.py`, …) is added to that rule's folder.

Ensures the backlog has sufficient open ideas (status `proposed` or `evaluated`). Each LLM call generates exactly one idea and is repeated until one of the following stop conditions is met:
- There are at least 3 open ideas **and** the highest-scored open idea has a score ≥ 0.6.
- 3 new ideas have been generated in this cycle (cap per run).

If both conditions are already satisfied at the start of the step, it is skipped entirely. New ideas are added with status `proposed`.

#### Step 7 — Evaluate ideas
*Inputs*: `long_term_plan.json`, `idea_backlog.json`  
*Output*: `idea_backlog.json` (scores and status updated)

In a single LLM call, scores **all** ideas in the backlog — both newly `proposed` and previously `evaluated` — against the long term plan, estimated feasibility, and potential signal quality. Re-evaluating existing ideas allows scores to be revised in light of updated conclusions and plan. Sets a score (0–1) for each idea, marks low-potential ones as `rejected`, and marks the rest as `evaluated`.

#### Step 8 — Select and implement one idea
*Inputs*: `idea_backlog.json`, `version_comparison.json`, registered rule versions from `strategy/rules/`  
*Output*: new or updated rule version file under `strategy/rules/`, `strategy.py` (registrations updated), `idea_backlog.json` (idea marked `implemented`)

Selects the highest-scored ready idea and generates the rule code:
- For a **new rule** idea: creates a new rule folder and `v1.py`, registers it in `strategy.py`.
- For a **modify rule** idea: creates the next version file in the existing rule's folder, registers it in `strategy.py` alongside the prior version.

At most one rule version is added per pipeline run to keep changes reviewable. Simultaneously, rule versions deprecated in step 2 and inferior versions marked for dropping in step 3 are unregistered from `strategy.py` (files are kept for signal traceability).

### 8.3 Rule lifecycle

```
[proposed in backlog] → [implemented / candidate] → [active] → [dropped]
                                                                   ▲
                                                       (version comparison
                                                        marks inferior versions)
```

- **Proposed**: idea exists in the backlog, not yet in `strategy/rules/`.
- **Candidate**: version file exists and is registered in `strategy.py`, but below the minimum signal threshold to be scored. Not eligible for version comparison.
- **Active**: version has enough signal history and composite score is above the deprecation threshold.
- **Dropped**: either score fell below the deprecation threshold, or version comparison identified it as significantly worse than another version of the same rule. Unregistered from `strategy.py` but the version file is kept for signal traceability.

---

## 9. LLM Interface

All model calls across the Strategy Updater pipeline are made through **LangChain**, which abstracts over providers and handles prompt templating. The underlying model is swappable via configuration without changing application code.

### 9.1 Model

The model name is read from `config.yaml` and passed to the LangChain chat model constructor at startup.

### 9.2 Structured output

All LLM responses that feed into application logic are parsed into **Pydantic models** using LangChain's `.with_structured_output()`. Each pipeline step has its own output model. Examples:

```python
class SignalEvaluation(BaseModel):
    overall_positive_rate: float
    avg_gain_24h: float
    avg_max_gain_24h: float
    notes: str

class RuleDescription(BaseModel):
    rule_id: str
    description: str

class RuleDescriptions(BaseModel):
    rules: list[RuleDescription]

class RuleScore(BaseModel):
    rule_id: str          # includes version, e.g. "rule_01_spread_compression_v2"
    description: str      # carried over from rule_descriptions.json
    signal_count: int
    avg_gain_24h: float
    positive_rate: float
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
    target_rule: str | None  # rule_name of the rule to modify; None for new_rule ideas
    score: float | None
    status: Literal["proposed", "evaluated", "implemented", "rejected"]

class IdeaBacklog(BaseModel):
    ideas: list[RuleIdea]

class ImplementedRule(BaseModel):
    idea_id: str
    rule_id: str
    code: str
```

---

## 10. Scheduler / Main Loop

The main process runs a cooperative loop with the following periodic tasks:

| Task | Trigger |
|---|---|
| Poll Kraken API + run strategy | Immediately after the previous cycle completes (rate-limit aware) |
| Downsample hot → warm tier | Every hour |
| Recompute cold-tier statistics | Every hour (after warm downsampling) |
| Evaluate pending signal outcomes | Every hour |
| Run Strategy Updater pipeline (all 8 steps) | Every 24 hours |

---

## 11. Configuration

A single `config.yaml` at the project root controls:

- Poll interval and backoff parameters
- Hot-tier max tick retention count
- Rule deprecation threshold and minimum signal count for scoring
- Data directory path
- State directory path (default: `data/state/`)
- LLM model name (e.g.: `gemini-2.0-flash`)
- Backtesting data directory (default: `../crypto_alerts_llm/data/raw`)

---

## 12. Open Questions / Future Work

- **Storage backend**: start with flat JSON files; migrate to SQLite if query performance degrades.
- **Signal deduplication**: if the same rule fires on consecutive ticks, subsequent signals from that rule for the same pair are suppressed for 24 hours.
- **Sell signals**: the current design is buy-only; a symmetric sell-signal path could be added later.
- **Alerting**: emit a notification (e.g., desktop notification, webhook) when a high-confidence buy signal fires.
- **Backtesting**: replay historical data through the strategy to evaluate rules without waiting for real-time outcomes. The backtesting module reads from an external data directory (configurable, default `../crypto_alerts_llm/data/raw`). That directory contains hourly snapshots partitioned as `year=YYYY/month=MM/day=DD/<timestamp>.json` (full Kraken Ticker response for all pairs) and `<timestamp>_bidask.json` (top-5 order book levels per pair). The backtester feeds these snapshots through the tiered storage layer and strategy engine in chronological order, producing a signal ledger that can be evaluated immediately since all future prices are available.
- **Multiple timeframes**: currently warm tier is hourly; finer granularity (e.g., 5-minute) could improve some rule types.
