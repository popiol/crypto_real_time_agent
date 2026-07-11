"""Rule 02 — Volatility-Adaptive Spread Compression Threshold."""
from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal


MIN_TICKS = 10
STD_DEV_MULTIPLIER = 1.5
RULE_ID = "0491eec1-5e56-4269-9669-9b503fc8b246"


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS:
            continue

        recent_spreads = [t.spread_rel for t in ticks]

        # Relative spread (spread_rel) is (ask - bid) / mid * 100.
        # For valid prices (ask > bid > 0), mid > 0, so spread_rel should always be positive.
        # If any spread is non-positive, it indicates malformed data, so we skip.
        if any(s <= 0 for s in recent_spreads):
            continue

        avg_relative_spread = statistics.mean(recent_spreads)

        # statistics.stdev requires at least two data points.
        # Since MIN_TICKS is 10, len(recent_spreads) will always be >= 10,
        # so a StatisticsError will not be raised for insufficient data points.
        # If all values are identical, stdev will be 0.0, which is handled correctly.
        std_dev_relative_spread = statistics.stdev(recent_spreads)

        # Define adaptive thresholds based on the average and standard deviation
        buy_threshold = avg_relative_spread - (STD_DEV_MULTIPLIER * std_dev_relative_spread)
        sell_threshold = avg_relative_spread + (STD_DEV_MULTIPLIER * std_dev_relative_spread)

        current_relative_spread = ticks[-1].spread_rel
        ts = ticks[-1].polled_at
        price = ticks[-1].last_price

        if current_relative_spread < buy_threshold:
            signals.append(BuySignal(
                pair=pair,
                timestamp=ts,
                price=price,
                rule_id=RULE_ID
            ))
        elif current_relative_spread > sell_threshold:
            signals.append(SellSignal(
                pair=pair,
                timestamp=ts,
                price=price,
                rule_id=RULE_ID
            ))

    return signals