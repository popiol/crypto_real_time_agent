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
| `data/state/signal_evaluation.json` | Aggregated per-rule signal statistics (win rate, avg gain, breakdown by pair and exit reason) |
| `data/state/rule_evaluation.json` | Per-rule-version scoring, status, description, and metrics; descriptions are cached here across runs |
| `data/state/version_comparison.json` | Results of comparing versions of the same rule; marks inferior versions for dropping |
| `data/state/conclusions.json` | Per-rule version direction conclusions: what the dropped modification tried, why it failed, what to try next |
| `data/state/long_term_plan.json` | Evolving high-level direction for the strategy |
| `data/state/idea_backlog.json` | Pool of rule ideas with status (`proposed`, `evaluated`, `implemented`, `rejected`) |

### 8.2 Pipeline steps

All steps that involve reasoning use the LLM (see section 9). Steps run sequentially; each reads its inputs from persisted files and writes its output before the next step begins.

#### Step 1 — Analyze results
*Inputs*: signal ledger (SQLite `signals` table, signals with filled-in outcomes)  
*Output*: `signal_evaluation.json`

Aggregates signal outcome data per rule: signal count, positive rate, average gain, breakdown by exit reason and by pair.

#### Step 2 — Analyze current rules
*Inputs*: signal ledger (directly from SQLite), prior `rule_evaluation.json` (description and zero-signal-cycle cache), registered rule versions from `strategy.py`  
*Output*: `rule_evaluation.json`

Processes each active rule version in two sub-steps:

1. **Describe** — for each version not already described in the prior `rule_evaluation.json`, makes one LLM call to produce a concise plain-language description. Descriptions are cached inside `rule_evaluation.json` and reused in subsequent runs.

2. **Score** — computes the following metrics from the signal ledger for each version:
   - `signal_count`, `evaluation_days`
   - `avg_gain_pct` — average gain across all evaluated signals
   - `recent_avg_gain_pct` — average gain across signals within the last 48 hours of the latest signal (data-relative, not wall-clock)
   - `positive_rate`, `avg_gain_24h`, `max_gain_24h`
   - `score` — composite score normalised to [0, 1]
   - `status` — `candidate` (insufficient signals), `active`, or `deprecate`
   - `zero_signal_cycles` — consecutive cycles with no signals emitted

#### Step 3 — Compare rule versions
*Inputs*: `rule_evaluation.json`  
*Output*: `version_comparison.json`

For each rule with more than one active version, compares composite scores. Versions significantly worse than the best-performing version of the same rule are marked for dropping. Versions in `candidate` status are excluded — they are not dropped until they have been scored. The output captures the rationale for each drop decision.

#### Step 4 — Derive conclusions
*Inputs*: `rule_evaluation.json`, `version_comparison.json`  
*Output*: `conclusions.json`

For each rule that has at least one version marked for dropping **and** at least one version being kept, makes a separate LLM call. The call receives the full version performance data (scores, descriptions) for that rule, the list of dropped versions, and the list of surviving versions. The LLM identifies what direction the dropped modification took, why it likely underperformed, and proposes a different direction to try next.

One LLM call is made per qualifying rule. Rules where all versions are dropped, or no versions are dropped, produce no conclusion.

#### Step 5 — Update long-term plan
*Inputs*: `conclusions.json`, `rule_evaluation.json`, existing `long_term_plan.json`  
*Output*: `long_term_plan.json` (updated)

Revises the strategic direction for the agent. The LLM receives both the full rule evaluation (scores and metrics for all active rules) and the version direction conclusions, so the updated plan reflects both overall rule health and specific lessons learned from version experiments. The plan is a persistent document that accumulates knowledge across many update cycles.

#### Step 6 — Generate ideas
*Inputs*: `long_term_plan.json`, `idea_backlog.json`, `rule_evaluation.json`  
*Output*: `idea_backlog.json` (new entries appended)

There are two kinds of ideas:
- **New rule** — proposes an entirely new rule concept. If implemented, a new rule folder and `v1.py` are created.
- **Modify rule** — proposes a **small, targeted change** to an existing rule (e.g. adjust a threshold, window length, or coefficient). Must not propose a structural rewrite — that should be a new rule instead. If implemented, a new version file (`v2.py`, `v3.py`, …) is added alongside the existing version.

Each LLM call generates exactly one idea and is repeated until one of the following stop conditions is met:
- There are at least 3 open ideas **and** the highest-scored open idea has a score ≥ 0.6.
- 3 new ideas have been generated in this cycle (cap per run).

If stop conditions are already met at the start, the step is skipped entirely.

#### Step 7 — Evaluate ideas
*Inputs*: `long_term_plan.json`, `idea_backlog.json`  
*Output*: `idea_backlog.json` (scores and status updated)

In a single LLM call, scores **all** ideas in the backlog — both newly `proposed` and previously `evaluated` — against the long-term plan, estimated feasibility, and potential signal quality. Re-evaluating existing ideas allows scores to be revised in light of updated conclusions and plan. Sets a score (0–1) for each idea, marks low-potential ones as `rejected`, and marks the rest as `evaluated`.

#### Step 8 — Select and implement one idea
*Inputs*: `idea_backlog.json`, `version_comparison.json`, source of the existing rule (for `modify_rule` ideas)  
*Output*: new or updated rule version file under `strategy/rules/`, `strategy.py` (registrations updated), `idea_backlog.json` (idea marked `implemented`)

Selects the highest-scored `evaluated` idea and generates real, executable Python code:
- For a **new rule** idea: creates a new rule folder and `v1.py`, registers it in `strategy.py`.
- For a **modify rule** idea: reads the existing rule source and generates the next version file with only the targeted change applied, registers it in `strategy.py` alongside the prior version.

At most one rule version is added per pipeline run. Simultaneously, rule versions deprecated in step 2 and inferior versions marked for dropping in step 3 are unregistered from `strategy.py` (files are kept for signal traceability). Both imports and `ACTIVE_RULES` entries are guarded for idempotency.

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

All LLM responses that feed into application logic are parsed into **Pydantic models**. Each pipeline step has its own output model:

```python
class RuleScore(BaseModel):
    rule_id: str                # includes version, e.g. "rule_01_spread_compression_v2"
    description: str            # cached from prior run or generated fresh
    signal_count: int
    evaluation_days: int
    avg_gain_pct: float
    recent_avg_gain_pct: float  # avg gain over signals in the last 48h of data
    positive_rate: float
    avg_gain_24h: float
    max_gain_24h: float
    score: float                # composite, normalised to [0, 1]
    status: Literal["candidate", "active", "deprecate"]
    zero_signal_cycles: int

class RuleEvaluation(BaseModel):
    rules: list[RuleScore]
    summary: str

class RuleVersionComparison(BaseModel):
    rule_name: str
    versions_compared: list[str]
    best_version: str
    versions_to_drop: list[str]
    rationale: str

class VersionComparisonResult(BaseModel):
    comparisons: list[RuleVersionComparison]
    summary: str

class VersionDirectionConclusion(BaseModel):
    rule_name: str
    dropped_versions: list[str]
    failed_direction: str    # what the dropped version tried
    proposed_direction: str  # different direction to explore next

class Conclusions(BaseModel):
    conclusions: list[VersionDirectionConclusion]

class LongTermPlan(BaseModel):
    direction: str
    priorities: list[str]
    updated_at: str

class RuleIdea(BaseModel):
    idea_id: str
    title: str
    description: str
    rationale: str
    pseudocode: str
    kind: Literal["new_rule", "modify_rule"]
    target_rule: str | None  # rule_name of the rule to modify; None for new_rule
    score: float | None
    status: Literal["proposed", "evaluated", "implemented", "rejected"]

class IdeaBacklog(BaseModel):
    ideas: list[RuleIdea]

class ImplementedRule(BaseModel):
    idea_id: str
    rule_id: str          # module name, e.g. "rule_13_new_concept"
    function_name: str    # Python function name
    code: str             # complete Python file content
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
    revenue: float      # net after fee
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

3. **Place new orders** — finds the rule with the highest `recent_avg_gain_pct` in `rule_evaluation.json`. If it exceeds `portfolio_min_recent_gain`, signals from that rule in the current cycle are acted on:
   - Buy signal → place a buy limit order at the signal price, spending `capital / 10`. Maximum 10 simultaneous open positions. Cash already committed to pending buy orders is deducted before checking available cash.
   - Sell signal → place a sell limit order at the signal price. If a sell order already exists for that position, it is cancelled and replaced.

4. **Update values** — all position `value` fields and `portfolio.value` are recomputed from current tick prices, then state is saved.

### 10.4 Configuration

| Field | Default | Meaning |
|---|---|---|
| `portfolio_initial_capital` | `10000.0` | Starting cash in USD |
| `portfolio_min_recent_gain` | `0.005` | Minimum `recent_avg_gain_pct` for a rule to trigger orders |
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
| Run Strategy Updater pipeline (all 8 steps) | Every 24 hours |

---

## 12. Configuration

A single `config.yaml` at the project root controls:

- Poll interval and backoff parameters
- Hot-tier max tick retention count
- Rule deprecation threshold and minimum signal count for scoring
- Data directory path (SQLite database + portfolio files live here)
- State directory path (default: `data/state/`)
- LLM model name (e.g.: `claude-cli`)
- Backtesting data directory (default: `../crypto_alerts_llm/data/raw`)
- Portfolio settings: `portfolio_initial_capital`, `portfolio_min_recent_gain`, `portfolio_fee`

---

## 13. Open Questions / Future Work

- **Signal deduplication**: if the same rule fires on consecutive ticks, subsequent signals from that rule for the same pair are suppressed for 24 hours.
- **Alerting**: emit a notification (e.g., desktop notification, webhook) when a high-confidence buy signal fires.
- **Backtesting**: replay historical data through the strategy to evaluate rules without waiting for real-time outcomes. The backtesting module reads from an external data directory (configurable, default `../crypto_alerts_llm/data/raw`). That directory contains hourly snapshots partitioned as `year=YYYY/month=MM/day=DD/<timestamp>.json` (full Kraken Ticker response for all pairs) and `<timestamp>_bidask.json` (top-5 order book levels per pair). The backtester feeds these snapshots through the tiered storage layer and strategy engine in chronological order, producing a signal ledger that can be evaluated immediately since all future prices are available.
- **Multiple timeframes**: currently warm tier is hourly; finer granularity (e.g., 5-minute) could improve some rule types.
