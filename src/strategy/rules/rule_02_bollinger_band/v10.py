"""Rule dbe0ee61-0918-4d1c-b7f2-8261e39eb84a — Bollinger Band Mean-Reversion with ATR-Filtered Entry."""
from __future__ import annotations

import numpy as np
from src.agent.models import BuySignal, MarketData, PairData, SellSignal

# Rule parameters
BB_PERIOD = 20
BB_STD_DEV = 2.0
ATR_PERIOD = 14
ATR_MULTIPLIER = 0.5

# Minimum number of warm candles required for calculations.
# Bollinger Bands need BB_PERIOD closes.
# ATR needs ATR_PERIOD True Ranges, which in turn requires ATR_PERIOD + 1 candles
# (because each True Range calculation uses the previous candle's close).
MIN_CANDLES = max(BB_PERIOD, ATR_PERIOD + 1)

# Unique identifier for this rule
RULE_ID = "dbe0ee61-0918-4d1c-b7f2-8261e39eb84a"


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candle data and at least one hot tick for current price/timestamp.
        # The `MIN_CANDLES` check covers both BB and ATR data requirements.
        if not pair_data.hot or len(pair_data.warm) < MIN_CANDLES:
            continue

        # Extract data for calculations, converting to numpy arrays for efficiency.
        closes = np.array([c.close for c in pair_data.warm])
        highs = np.array([c.high for c in pair_data.warm])
        lows = np.array([c.low for c in pair_data.warm])

        current_price = pair_data.hot[-1].last_price
        ts = pair_data.hot[-1].polled_at

        # 1. Calculate Bollinger Bands
        # Use the last BB_PERIOD closing prices for SMA and STDDEV.
        bb_closes = closes[-BB_PERIOD:]
        mid_band = np.mean(bb_closes)
        std_dev = np.std(bb_closes)

        # Avoid division by zero or meaningless bands if standard deviation is zero.
        if std_dev == 0:
            continue

        upper_band = mid_band + (std_dev * BB_STD_DEV)
        lower_band = mid_band - (std_dev * BB_STD_DEV)

        # 2. Calculate Average True Range (ATR)
        # True Range (TR) for each candle 't' is:
        # TR_t = max((high_t - low_t), abs(high_t - close_{t-1}), abs(low_t - close_{t-1}))

        # To calculate TR for the last `N-1` candles, we need `N` candles.
        # `highs[1:]` corresponds to H_t, `lows[1:]` to L_t, `closes[:-1]` to C_{t-1}.
        hl = highs[1:] - lows[1:]  # High minus Low
        hc = np.abs(highs[1:] - closes[:-1])  # High minus Previous Close
        lc = np.abs(lows[1:] - closes[:-1])  # Low minus Previous Close

        # Array of True Range values for each period (length `len(closes) - 1`).
        true_ranges = np.maximum(hl, np.maximum(hc, lc))

        # Calculate ATR as the simple moving average of the last ATR_PERIOD True Ranges.
        # This is a common interpretation when a specific smoothing method (like exponential)
        # is not explicitly defined in the pseudocode.
        atr_value = np.mean(true_ranges[-ATR_PERIOD:])

        # 3. Calculate ATR-Filtered Buy/Sell Thresholds
        # These thresholds extend the Bollinger Bands by a multiple of the ATR.
        buy_threshold = lower_band - (atr_value * ATR_MULTIPLIER)
        sell_threshold = upper_band + (atr_value * ATR_MULTIPLIER)

        # 4. Generate Signals
        # A Buy signal is emitted if the current price drops below the extended lower threshold.
        if current_price <= buy_threshold:
            signals.append(BuySignal(
                pair=pair,
                timestamp=ts,
                price=current_price,
                rule_id=RULE_ID
            ))
        # A Sell signal is emitted if the current price rises above the extended upper threshold.
        elif current_price >= sell_threshold:
            signals.append(SellSignal(
                pair=pair,
                timestamp=ts,
                price=current_price,
                rule_id=RULE_ID
            ))

    return signals