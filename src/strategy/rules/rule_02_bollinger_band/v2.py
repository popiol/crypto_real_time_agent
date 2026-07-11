from __future__ import annotations

import statistics

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle


K = 2.0
MIN_CANDLES = 10  # Minimum candles for Bollinger Band calculation


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure enough warm candles for Bollinger Band calculation AND 2-candle confirmation.
        # We need at least MIN_CANDLES for the Bollinger Band calculation (mean and std dev).
        # We also need at least 2 candles (warm[-1] and warm[-2]) to get current and previous closes.
        # Since MIN_CANDLES is 10, this check implicitly covers needing at least 2 candles.
        if len(pair_data.warm) < MIN_CANDLES:
            continue

        # Ensure we have at least one hot tick for the signal timestamp and price.
        if not pair_data.hot:
            continue

        # Extract close prices from all available warm candles for Bollinger Band calculation.
        closes = [c.close for c in pair_data.warm]

        # Calculate Bollinger Band components.
        # `statistics.stdev` requires at least 2 data points. Our MIN_CANDLES (10) ensures this.
        mean = statistics.mean(closes)

        # Handle cases where standard deviation cannot be calculated (e.g., all closes are the same)
        # or if there are fewer than 2 data points (though MIN_CANDLES should prevent this).
        if len(closes) < 2:
            continue
        std = statistics.stdev(closes)

        if std == 0:  # Avoid division by zero or bands collapsing to a single line
            continue

        upper_band = mean + K * std
        lower_band = mean - K * std

        # Get the close prices of the current and previous candles for confirmation.
        # These indices are safe because len(pair_data.warm) >= MIN_CANDLES >= 2.
        current_candle: WarmCandle = pair_data.warm[-1]
        previous_candle: WarmCandle = pair_data.warm[-2]

        current_close = current_candle.close
        previous_close = previous_candle.close

        # Get the timestamp and price from the latest hot tick for the signal.
        # This reflects the exact moment the signal is generated based on current market data.
        latest_tick: Tick = pair_data.hot[-1]
        signal_timestamp = latest_tick.polled_at
        signal_price = latest_tick.last_price

        # Apply the two-candle confirmation logic for Buy and Sell signals.
        if current_close < lower_band and previous_close < lower_band:
            signals.append(BuySignal(pair=pair, timestamp=signal_timestamp, price=signal_price))
        elif current_close > upper_band and previous_close > upper_band:
            signals.append(SellSignal(pair=pair, timestamp=signal_timestamp, price=signal_price))

    return signals