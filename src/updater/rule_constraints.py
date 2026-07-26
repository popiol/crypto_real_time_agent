"""Shared hard-constraint prompt text for anything that asks an LLM to design or
implement a trading rule — idea generation (step5), code generation/fixing
(step6), and failure diagnosis (plan_next_cycle). Kept in their own module
(rather than defined in step6_implement_rule.py and imported from there) so
plan_next_cycle.py can use them without an import cycle, since step6 already
imports plan_next_cycle.
"""

from __future__ import annotations

DATA_WINDOW_CONSTRAINT = (
    "DATA TIER LIMITS (hard constraints — a rule that violates these will never "
    "produce a signal against real data, even though it will look syntactically valid):\n"
    "- data.hot: the last ~300 raw ticks (~5 minutes of history at 1 poll/sec). "
    "Tick-level last_price/bid/ask/spread.\n"
    "- data.warm: the last 24 hourly OHLC candles, AT MOST — never more than 24 entries. "
    "Any indicator whose lookback exceeds 24 hourly candles (e.g. SMA(30), SMA(50), or "
    "any period > 23) CANNOT be computed from data.warm; len(data.warm) will never reach "
    "that many entries, so the rule will always fall through its own 'insufficient data' "
    "check and return [].\n"
    "- data.cold: ONE ROW PER CALENDAR MONTH, aggregates only (min_price, max_price, "
    "avg_price, avg_daily_spread, candle_count, last_candle_hour) — it is NOT an hourly "
    "or daily price series. It cannot be used to extend a warm-tier indicator's lookback "
    "(there is no way to reconstruct individual hourly/daily closes from it). Only use "
    "data.cold for coarse, monthly-resolution comparisons (e.g. current price vs. this "
    "month's avg_price/min_price/max_price).\n"
    "Every indicator lookback period MUST fit within data.warm's 24-candle limit (or use "
    "data.hot/data.cold directly). Do not design an indicator assuming a longer hourly or "
    "daily history exists anywhere — it does not."
)

LONG_ONLY_CONSTRAINT = (
    "LONG-ONLY (hard constraint): this system only ever holds long positions. "
    "BuySignal opens one; SellSignal must ONLY be used to close an existing long "
    "position for that pair — never as a standalone bearish/short entry hypothesis. "
    "A rule whose core hypothesis is 'detect a bearish/overbought/breakdown setup and "
    "emit SellSignal to open a short' will never be evaluated: nothing in this system "
    "resolves outcomes for a sell-direction entry, so its signals accumulate forever "
    "with no result. If the hypothesis is fundamentally bearish (expecting a price "
    "decline), express it as a SellSignal that exits a long entered by an *earlier* "
    "bullish condition in the same rule — not as the rule's only/primary signal."
)

STATELESS_CONSTRAINT = (
    "NO POSITION VISIBILITY (hard constraint): signal(data: MarketData) receives ONLY "
    "market data — data is dict[str, PairData], nothing else. There is no data.positions, "
    "data.portfolio, data.holdings, or any other way for a rule to see which pairs it "
    "currently holds, an entry price, or how long a position has been open. Do NOT write "
    "exit logic that depends on entry price (e.g. 'sell once price has dropped 2% from "
    "where I bought' or 'sell once profit exceeds 3% from entry') — that state does not "
    "exist inside signal() and any attempt to access it (a .positions attribute, a passed-"
    "in argument, a module-level variable assumed to persist between calls) will crash or "
    "silently do nothing, since signal() is a pure function called fresh each tick with no "
    "memory of prior calls. Express SellSignal conditions using only what's visible in "
    "`data` itself — a market-derived reversal condition, an indicator crossing back, a "
    "time/candle-count condition — never a percentage move relative to an assumed entry "
    "price. (The system matches a SellSignal to an open position purely by pair; it does "
    "not need or use an entry price from the rule.)"
)

SIGNAL_FIELDS_CONSTRAINT = (
    "SIGNAL FIELDS (hard constraint): BuySignal/SellSignal have exactly these fields "
    "you may set: pair, timestamp, price, rule_id, confidence. Do NOT pass `indicators=` "
    "yourself — it is filled in automatically, after your signal() function returns, by "
    "the code that calls it; any value you set will be silently overwritten. Do NOT "
    "invent other fields (e.g. profit_target_pct, stop_loss_pct, max_hold_hours) to "
    "communicate exit parameters to 'the trading agent' — no such mechanism exists. "
    "Position sizing and exit timing are fixed and config-driven (equal-sized positions, "
    "a fixed max hold time), not influenced by anything a rule returns beyond pair/price/"
    "timestamp/confidence. A rule that tries to pass extra per-signal parameters this way "
    "will crash with a validation error the moment it fires, since indicators must be "
    "dict[str, float | None] — no strings, no other keys."
)
