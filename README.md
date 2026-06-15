# Crypto Real-Time Agent

An AI-powered agent that monitors cryptocurrency markets in real time, identifies buy signals using an evolving rule-based strategy, and tracks the performance of each rule over time.

## What it does

- **Pulls live market data** from the Kraken API as frequently as rate limits allow, including current quotes and bid/ask spreads.
- **Stores historical data** in local files with granularity that decreases with age: full resolution for the most recent data, hourly prices for the last 24 hours, and statistical summaries (monthly min/max, average daily spread) for older data.
- **Applies a trading strategy** implemented as a Python script that analyzes all available data and emits buy signals.
- **Tracks buy signals** along with the rule that triggered each one, then evaluates each signal by measuring the maximum gain achievable within the next 24 hours and the final gain at the 24-hour mark.
- **Evolves the strategy** by analyzing the accumulated performance data for each rule — rules that consistently fail to produce profitable signals are removed, and new candidate rules are introduced.

## Key concepts

| Concept | Description |
|---|---|
| **Data collection** | Real-time polling of Kraken REST API with adaptive rate-limit handling |
| **Tiered storage** | Recent ticks → hourly aggregates → monthly statistics |
| **Strategy script** | A standalone Python module that ingests stored data and returns buy signals |
| **Signal ledger** | A persistent record of every buy signal, the rule that fired it, and its eventual outcome |
| **Rule lifecycle** | Rules are introduced, tested against live signals, scored, and retired when underperforming |

## Documentation

See [docs/design.md](docs/design.md) for the full architecture and design decisions.
