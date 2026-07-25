from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal

# Constants for indicator calculations
N_PERIOD = 20  # Period for SMA and STD for both indicators, using warm candles
BB_K = 2.0     # Standard deviation multiplier for Bollinger Bands calculation

# Rule thresholds
OSCILLATOR_THRESHOLD = -1.5
BB_WIDTH_THRESHOLD = 0.8 # Note: this is a dimensionless ratio, typically for BB width it's a percentage (e.g. 0.05 for 5%), 0.8 is a very high value, implying extreme volatility.

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough warm candle data for calculations
        # We need at least N_PERIOD candles to calculate SMA and STD for the last candle.
        if len(warm_candles) < N_PERIOD:
            continue

        # Extract close prices for the last N_PERIOD warm candles
        # These are used to calculate the SMA and STD that define the "warm" state.
        relevant_closes = np.array([c.close for c in warm_candles[-N_PERIOD:]])

        # Calculate Simple Moving Average (SMA) and Standard Deviation (STD)
        # These form the basis for both the Mean Reversion Oscillator and Bollinger Band Width.
        sma = np.mean(relevant_closes)
        std = np.std(relevant_closes)

        # Handle cases where std or sma might be zero to prevent division errors
        # A zero standard deviation means no price movement, which is not volatile.
        # A zero SMA is unlikely for price data but would also cause division by zero.
        if std == 0 or sma == 0:
            continue

        # The most recent close price for the oscillator calculation
        current_close = warm_candles[-1].close

        # 1. Calculate Warm Mean Reversion Oscillator
        # This measures how far the current price is from its mean, normalized by volatility.
        # A negative value indicates the price is below the mean.
        warm_mean_reversion_oscillator = (current_close - sma) / std

        # 2. Calculate Warm Bollinger Band Width
        # This measures the width of the Bollinger Bands relative to the middle band (SMA),
        # indicating market volatility.
        # BB Width = (Upper Band - Lower Band) / Middle Band = ( (SMA + K*STD) - (SMA - K*STD) ) / SMA
        # = (2 * K * STD) / SMA
        warm_bollinger_band_width = (2 * BB_K * std) / sma

        # Apply the trading rule condition:
        # Initiate a long position if the oscillator signals a deep dip AND
        # the Bollinger Band Width confirms high volatility.
        if (warm_mean_reversion_oscillator < OSCILLATOR_THRESHOLD and
                warm_bollinger_band_width >= BB_WIDTH_THRESHOLD):

            # Determine the price and timestamp for the signal.
            # Prioritize the latest real-time tick data if available for precision.
            # Otherwise, fall back to the close price and hour of the last warm candle.
            signal_price = current_close
            signal_timestamp = warm_candles[-1].hour # Use candle hour if no hot data
            
            if pair_data.hot:
                latest_tick = pair_data.hot[-1]
                signal_price = latest_tick.last_price
                signal_timestamp = latest_tick.polled_at

            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
                rule_id="DeepVolatileDipBuy",
                confidence=1.0 # High conviction signal
            ))

    return signals