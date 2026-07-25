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
def find_signals(data: MarketData) -> list[BuySignal | SellSignal]:
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

Exactly one rule is active at a time. `strategy.py` holds a single `ACTIVE_RULE` import, and `find_signals()` calls only that module's `signal()` function — this matches the Strategy Updater's one-hypothesis-per-cycle learning loop (§8): there is always exactly one hypothesis under test, never a portfolio of concurrently running rules. Rules are purely functional — they read data and return signals; they have no side effects.

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

Each rule version has a unique `rule_id` formed from the rule name and version (e.g., `rule_01_spread_compression_v2`). When the Strategy Updater implements a new rule idea (§8.2 Step 5), it replaces `ACTIVE_RULE` in `strategy.py` with the new version — a `fix` idea's version file is added alongside the previous version of the same rule, a `new_rule` idea's file goes in a new folder. Either way, the new version immediately becomes the sole active rule. The previous version's file is kept on disk under `strategy/rules/` for signal traceability but is no longer imported or executed once replaced.

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

| File / Directory | Contents |
|---|---|
| `data/state/signal_evaluation.json` | Aggregated per-rule signal statistics (win rate, avg gain, breakdown by pair and exit reason) |
| `data/state/rule_evaluation.json` | Accumulated per-rule-version scoring, description, and enhanced metrics — one entry per rule ever evaluated; descriptions are cached here across runs |
| `data/state/indicator_set.json` | The current set of LLM-defined indicators: name, description, and generated Python code for each |
| `data/state/train_set.json` | Accumulating samples of (indicator values from 24h ago, target: last 24h change) across all pairs and cycles |
| `data/state/traces/` | Episodic trace store — one JSON file per cycle, immutable; contains hypothesis, indicator set version, outcome metrics, and LLM diagnosis |
| `data/state/next_cycle_plan.json` | Plan produced at end of each cycle: action (`continue`, `fix`, `new_rule`), and for `fix`/`new_rule`, a description of what to attempt |
| `data/state/last_implemented.json` | `{rule_id, cycle_id}` of the most recently implemented rule (the current `ACTIVE_RULE`). Signal outcomes lag implementation by design (24h+ to resolve), so this is how later steps in the *following* cycle (evaluate, trace, plan, and a `fix` idea's `target_rule`) know which rule they're following up on |

### 8.2 Pipeline steps

All steps that involve reasoning use the LLM (see section 9). Steps run sequentially; each reads its inputs from persisted files and writes its output before the next step begins. The pipeline runs every 24 hours.

#### Step 1 — Update indicator set
*Inputs*: `data/state/next_cycle_plan.json`, `data/state/indicator_set.json`  
*Output*: `data/state/indicator_set.json` (updated)

Skipped if `next_cycle_plan.json` has `action: continue` (previous cycle was successful — gain > 0.5%). Otherwise, the LLM reviews the current indicator set in light of the plan and may add, remove, or modify indicators. For each change, the LLM generates a Python function that computes the indicator from `PairData` (hot/warm/cold tiers). The updated set is persisted.

#### Step 2 — Compute indicators
*Inputs*: `data/state/indicator_set.json`, hot/warm/cold tier data for all pairs (snapshot from 24h ago)  
*Output*: in-memory indicator values per pair (consumed immediately by step 3)

Executes each indicator function against the tier snapshot. Results are a flat dict of `{indicator_name: value}` per pair.

#### Step 3 — Relation analysis
*Inputs*: indicator values (from step 2), `data/state/train_set.json`, top-K episodic traces retrieved from `data/state/traces/` by semantic similarity to the current plan  
*Output*: in-memory analysis (consumed by step 4)

The LLM receives: the computed indicators for all pairs, the target values (last 24h change per pair), the accumulated train set, and the most relevant past traces. It identifies which indicator patterns preceded positive and negative outcomes, and what those relations suggest about the next rule to try.

Traces are retrieved by embedding the current plan and performing cosine similarity search over trace hypothesis embeddings. Top-K (default 5) most similar traces are included.

#### Step 4 — Generate rule idea
*Inputs*: relation analysis (from step 3), `data/state/next_cycle_plan.json`  
*Output*: selected idea (in-memory)

The LLM generates exactly one idea per cycle, derived from the relation analysis. Two kinds:
- **New rule** — an entirely new rule concept; if implemented, a new rule folder and `v1.py` are created. `target_rule` is null.
- **Fix** — a targeted change to the currently active rule; if implemented, a new version file is added alongside the existing one. `target_rule` is always the current `ACTIVE_RULE`'s `rule_id`, set directly from `last_implemented.json` rather than asked of the LLM — since only one rule is ever active, there is never any ambiguity about what a "fix" targets.

#### Step 5 — Implement rule
*Inputs*: selected idea, source of the existing rule (for fix ideas)  
*Output*: new rule version file under `strategy/rules/`, `strategy.py` (`ACTIVE_RULE` updated)

Generates real, executable Python code from the idea produced in step 4. Exactly one rule version is added per pipeline run, and it becomes the sole `ACTIVE_RULE` in `strategy.py`, replacing whichever rule was previously active. The previous version's file is kept under `strategy/rules/` for signal traceability but is no longer imported or executed.

#### Step 6 — Evaluate outcomes and score the active rule
*Inputs*: signal ledger (signals emitted more than 24h ago with `outcome = null`), prior `rule_evaluation.json`  
*Output*: `rule_evaluation.json` (updated — one entry, for the current `ACTIVE_RULE`), `signal_evaluation.json`

Fills in signal outcomes from tier data. Computes metrics for the currently active rule:
- `signal_count`, `evaluation_days`
- `avg_gain_pct`, `recent_avg_gain_pct` (last 48h of data)
- `min_gain_pct` (worst result), `p25_gain_pct`, `p75_gain_pct`
- `positive_rate`, `avg_gain_24h`, `max_gain_24h`
- `longest_win_streak`, `longest_loss_streak`
- `weekly_signal_counts` — list of signal counts per week, oldest first
- `signal_trend` — `increasing`, `decreasing`, or `stable` (derived from weekly counts)
- `avg_gain_by_volatility` — avg gain split into low/medium/high Bollinger Band Width buckets at signal time
- `score` — composite score normalised to [0, 1]

`rule_evaluation.json` accumulates one entry per rule ever evaluated: the active rule's entry is refreshed each cycle; every other rule_id's entry is carried over unchanged from whenever it was last active. There is no `status` field — whether to keep or replace the active rule is decided purely from `recent_avg_gain_pct + avg_transaction_gain` (§8.2 Step 9), not from a separate classification.

#### Step 7 — Update train set
*Inputs*: indicator values computed in step 2, target values (last 24h change per pair)  
*Output*: `data/state/train_set.json` (new samples appended)

Appends one record per pair: `{cycle_id, pair, indicators: {...}, target_gain_pct}`. The train set accumulates indefinitely across cycles.

#### Step 8 — Write episodic trace
*Inputs*: selected idea (from step 4), `rule_evaluation.json`, relation analysis diagnosis  
*Output*: new file in `data/state/traces/<cycle_id>.json`

Writes an immutable trace record:
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

Traces are never edited. The `embedding` field is computed from the hypothesis text and stored alongside the trace for semantic retrieval in future cycles.

#### Step 9 — Plan next cycle
*Inputs*: `rule_evaluation.json` (score for the currently active rule), episodic trace (from step 8)  
*Output*: `data/state/next_cycle_plan.json`

Decision logic, evaluated fresh every cycle against whichever rule is currently active. The metric judged depends on `transaction_count`, the number of transactions the portfolio has actually closed for the rule:
- Below 10 transactions: `recent_avg_gain_pct + avg_transaction_gain` — the same combined formula as the portfolio's own trading gate (§10.3). With few real trades, blending in the signal-theoretical figure gives a less noisy read; it degrades gracefully to just `recent_avg_gain_pct` while `transaction_count` is `0`, since `avg_transaction_gain` is `0.0` by construction until then.
- At 10 or more: `avg_transaction_gain` alone. With enough real trades to be a trustworthy sample, the realized result is trusted exclusively — a rule with a rosy theoretical `recent_avg_gain_pct` but real losses no longer gets a pass.
- If that metric > 0.5% (or the rule has too few signals to score yet): `action: continue` — leave the indicator set and the active rule alone.
- Otherwise: LLM attempts to diagnose whether the failure is fixable (wrong thresholds, wrong indicators) or the hypothesis itself was wrong.
  - If fixable: `action: fix` with a description of the specific change to attempt.
  - If not fixable: `action: new_rule` — relation analysis starts fresh from the data.

If a cycle reads `action: continue` from the previous cycle, it re-runs this check immediately rather than trusting the stale verdict: if the rule is still performing, the cycle ends there; if it's no longer performing, the cycle proceeds straight into relation analysis → idea → implementation in the same run, rather than waiting a full cycle to act. Whenever a rule is actually replaced this cycle, the plan written for the *next* cycle is always `continue` for the new rule — the old rule's diagnosis is never carried over and misapplied to its replacement, since the new rule hasn't had any chance yet to earn a verdict of its own.

### 8.3 Rule lifecycle

```
[idea generated] → [implemented] → [continue | replaced]
```

Ideas are generated in step 4 and implemented immediately in step 5 — they are not queued or persisted separately; there is no backlog. Once implemented, a rule stays active for as long as step 9 keeps deciding `continue`. There is no separate status classification or grace-period counter: replacement is driven purely by `avg_gain_pct` against `cycle_success_threshold`, re-checked every cycle. A replaced rule's file remains under `strategy/rules/` for signal traceability, but is no longer imported or executed once `ACTIVE_RULE` (§5.3) points elsewhere.

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

class TrainSample(BaseModel):
    cycle_id: str
    pair: str
    indicators: dict[str, float | None]
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
    signal_count: int
    evaluation_days: int
    avg_gain_pct: float                 # theoretical, from signal entry/exit prices
    recent_avg_gain_pct: float          # avg gain over signals in the last 48h of data
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
    title: str
    description: str
    rationale: str
    pseudocode: str
    kind: Literal["new_rule", "fix"]
    target_rule: str | None     # rule_name of the rule to fix; None for new_rule

class NextCyclePlan(BaseModel):
    action: Literal["continue", "fix", "new_rule"]
    description: str | None     # what to attempt; None when action is "continue"

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

3. **Place new orders** — looks up the active rule (`last_implemented.json`) in `rule_evaluation.json`. There is exactly one active rule at a time (§5.3), so this never picks among candidates — it only checks whether that one rule's `recent_avg_gain_pct + avg_transaction_gain` exceeds `portfolio_min_recent_gain`. If it does, signals from that rule in the current cycle are acted on:
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
