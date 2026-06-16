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
                         │   Rule Analyzer     │
                         │  (prune / evolve)   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   LLM Interface     │
                         │  (LangChain)        │
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

Each rule is a self-contained function within `strategy.py` with the signature:

```python
def rule_<id>(data: MarketData) -> list[BuySignal]:
    ...
```

`find_buy_signals` calls all registered rules and merges their outputs. Rules are purely functional — they read data and return signals; they have no side effects.

### 5.3 Rule versioning

Each rule has a unique string `rule_id` (e.g., `spread_compression_v1`). When a rule is revised rather than replaced, a new `_v2` variant is added and the old one is deprecated (kept in the file but not registered) to preserve traceability of historical signals.

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

## 8. Rule Analyzer

A periodic (e.g., daily) analysis job reads the completed signal records and produces a per-rule performance summary:

| Metric | Description |
|---|---|
| `signal_count` | Total signals emitted |
| `avg_gain_24h` | Average final gain after 24 h |
| `avg_max_gain_24h` | Average peak gain available within 24 h |
| `positive_rate` | Fraction of signals with `gain_24h > 0` |
| `score` | Composite score (TBD formula) |

### 8.1 Rule lifecycle

```
[candidate] → [active] → [deprecated] → [removed]
```

- **Candidate**: newly added rule, not yet scored (fewer than N signals, configurable).
- **Active**: rule with enough signal history; remains active while score is above threshold.
- **Deprecated**: score fell below threshold; no longer called by `find_buy_signals`, but kept in the module for traceability.
- **Removed**: manually pruned from the module after review.

New rules are introduced manually by editing `strategy.py`. The threshold for automatic deprecation is configurable.

---

## 9. LLM Interface

The agent uses a large language model for tasks that benefit from natural language reasoning: interpreting rule performance summaries, suggesting new rule candidates, and explaining why signals were or were not profitable.

### 9.1 Framework

All model calls are made through **LangChain**, which abstracts over model providers and handles prompt templating. This makes the underlying model swappable via configuration without changing application code.

### 9.2 Model

Default model: `gemini-2.0-flash` (via `langchain-google-genai`). The model name is read from `config.yaml` and passed to the LangChain chat model constructor at startup.

### 9.3 Structured output

All LLM responses that feed into application logic are parsed into **Pydantic models** using LangChain's `.with_structured_output()` method. This ensures type safety and eliminates ad-hoc string parsing. Example output models:

```python
class RuleSuggestion(BaseModel):
    rule_id: str
    description: str
    rationale: str
    suggested_code: str

class PerformanceInterpretation(BaseModel):
    summary: str
    rules_to_deprecate: list[str]
    rules_to_investigate: list[str]
```

Free-form outputs (e.g., explanations shown to the user) are returned as plain strings and are not parsed.

---

## 10. Scheduler / Main Loop

The main process runs a cooperative loop with the following periodic tasks:

| Task | Trigger |
|---|---|
| Poll Kraken API + run strategy | Immediately after the previous cycle completes (rate-limit aware) |
| Downsample hot → warm tier | Every hour |
| Recompute cold-tier statistics | Every hour (after warm downsampling) |
| Evaluate pending signal outcomes | Every hour |
| Analyze rule performance | Every 24 hours |

---

## 11. Configuration

A single `config.yaml` at the project root controls:

- Tracked pairs (e.g., `["XBTUSD", "ETHUSD"]`)
- Poll interval and backoff parameters
- Hot-tier max tick retention count
- Rule deprecation threshold and minimum signal count
- Data directory path
- LLM model name (default: `gemini-2.0-flash`)
- Backtesting data directory (default: `../crypto_alerts_llm/data/raw`)

---

## 12. Open Questions / Future Work

- **Storage backend**: start with flat JSON files; migrate to SQLite if query performance degrades.
- **Signal deduplication**: if the same rule fires on consecutive ticks, subsequent signals from that rule for the same pair are suppressed for 24 hours.
- **Sell signals**: the current design is buy-only; a symmetric sell-signal path could be added later.
- **Alerting**: emit a notification (e.g., desktop notification, webhook) when a high-confidence buy signal fires.
- **Backtesting**: replay historical data through the strategy to evaluate rules without waiting for real-time outcomes. The backtesting module reads from an external data directory (configurable, default `../crypto_alerts_llm/data/raw`). That directory contains hourly snapshots partitioned as `year=YYYY/month=MM/day=DD/<timestamp>.json` (full Kraken Ticker response for all pairs) and `<timestamp>_bidask.json` (top-5 order book levels per pair). The backtester feeds these snapshots through the tiered storage layer and strategy engine in chronological order, producing a signal ledger that can be evaluated immediately since all future prices are available.
- **Multiple timeframes**: currently warm tier is hourly; finer granularity (e.g., 5-minute) could improve some rule types.
