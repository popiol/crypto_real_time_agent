from __future__ import annotations
import numpy as np
import statistics # Kept as per "Import only" instruction, though not strictly used for this rule's calculations.
from src.agent.models import BuySignal, SellSignal, PairData

RULE_ID = "rule_02_bollinger_band_v2"

# Rule parameters
LOOKBACK_PERIOD_BB = 20
STD_DEV_MULTIPLIER = 2
LOOKBACK_PERIOD_RSI = 14
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70

# Minimum ticks required for calculations:
# Bollinger Bands need LOOKBACK_PERIOD_BB ticks.
# RSI needs LOOKBACK_PERIOD_RSI + 1 ticks (to calculate LOOKBACK_PERIOD_RSI price changes).
MIN_TICKS = max(LOOKBACK_PERIOD_BB, LOOKBACK_PERIOD_RSI + 1)

MarketData = dict[str, PairData]

def _calculate_sma(prices: np.ndarray, period: int) -> float:
    """Calculates the Simple Moving Average (SMA) for the last 'period' prices."""
    if len(prices) < period:
        return np.nan
    return np.mean(prices[-period:])

def _calculate_std_dev(prices: np.ndarray, period: int) -> float:
    """Calculates the Standard Deviation for the last 'period' prices."""
    if len(prices) < period:
        return np.nan
    return np.std(prices[-period:])

def _calculate_rsi(prices: np.ndarray, period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for the given prices.
    Returns the latest RSI value.
    """
    # RSI requires at least period + 1 prices to calculate the first 'period' deltas
    if len(prices) <= period:
        return np.nan

    # Calculate price changes (deltas)
    # diff will have len(prices) - 1 elements
    diff = np.diff(prices)

    # Separate gains and losses
    gain = np.maximum(0, diff)
    loss = np.maximum(0, -diff)

    # Initialize arrays for smoothed averages, starting from the first `period` of deltas
    # These arrays will store the smoothed averages for each point after the initial period.
    avg_gains = np.zeros(len(diff))
    avg_losses = np.zeros(len(diff))

    # Calculate initial average gain and loss (simple average over the first `period` changes)
    # The first valid smoothed average is at index `period-1` in `avg_gains`/`avg_losses`
    # because it corresponds to the `period`-th price change.
    avg_gains[period-1] = np.mean(gain[:period])
    avg_losses[period-1] = np.mean(loss[:period])

    # Calculate subsequent smoothed averages using Wilder's smoothing method
    # avg_gain_n = ((prev_avg_gain * (period - 1)) + current_gain) / period
    for i in range(period, len(diff)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gain[i]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + loss[i]) / period

    # Get the last calculated average gain and loss
    final_avg_gain = avg_gains[-1]
    final_avg_loss = avg_losses[-1]

    # Calculate Relative Strength (RS)
    if final_avg_loss == 0:
        # If no losses, RS is infinite (unless no gains either, then 0 to avoid division by zero issues)
        rs = np.inf if final_avg_gain > 0 else 0.0
    else:
        rs = final_avg_gain / final_avg_loss

    # Calculate Relative Strength Index (RSI)
    if rs == np.inf:
        rsi = 100.0
    else:
        rsi = 100 - (100 / (1 + rs))

    return rsi

def bollinger_band_v2(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Enhanced Bollinger Band with RSI Confirmation trading rule.
    Generates buy signals when price falls below the lower Bollinger Band AND RSI is oversold.
    Generates sell signals when price rises above the upper Bollinger Band AND RSI is overbought.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Extract closing prices from the historical ticks
        prices = np.array([t.last_price for t in pair_data.hot])

        # Ensure we have enough data for calculations
        if len(prices) < MIN_TICKS:
            continue

        current_price = prices[-1]

        # Calculate Bollinger Bands
        sma_bb = _calculate_sma(prices, LOOKBACK_PERIOD_BB)
        std_bb = _calculate_std_dev(prices, LOOKBACK_PERIOD_BB)

        # Skip if Bollinger Band indicators cannot be computed (e.g., due to NaN values from insufficient data)
        if np.isnan(sma_bb) or np.isnan(std_bb):
            continue

        upper_band = sma_bb + (STD_DEV_MULTIPLIER * std_bb)
        lower_band = sma_bb - (STD_DEV_MULTIPLIER * std_bb)

        # Calculate RSI
        rsi = _calculate_rsi(prices, LOOKBACK_PERIOD_RSI)

        # Skip if RSI cannot be computed
        if np.isnan(rsi):
            continue

        # Generate Signals
        # Buy signal: Price below lower band AND RSI oversold
        if current_price < lower_band and rsi < RSI_OVERSOLD_THRESHOLD:
            signals.append(BuySignal(
                pair=pair,
                rule_id=RULE_ID,
                timestamp=pair_data.hot[-1].polled_at,
                price=current_price,
            ))
        # Sell signal: Price above upper band AND RSI overbought
        elif current_price > upper_band and rsi > RSI_OVERBOUGHT_THRESHOLD:
            signals.append(SellSignal(
                pair=pair,
                rule_id=RULE_ID,
                timestamp=pair_data.hot[-1].polled_at,
                price=current_price,
            ))

    return signals