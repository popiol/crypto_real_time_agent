from datetime import datetime
import numpy as np
from src.agent.models import BuySignal, SellSignal, MarketData, WarmCandle

# Constants for the rule
RSI_PERIOD = 14
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70
# Need at least RSI_PERIOD + 1 candles to calculate the first RSI value.
# Need at least RSI_PERIOD + 2 candles to calculate current AND previous RSI values for cross detection.
MIN_CANDLES_FOR_RSI = RSI_PERIOD + 2


def _calculate_rsi(prices: list[float], period: int) -> list[float]:
    """
    Calculates the Relative Strength Index (RSI) for a given list of prices.

    Args:
        prices: A list of close prices.
        period: The lookback period for RSI calculation.

    Returns:
        A list of RSI values. Returns an empty list if insufficient data.
    """
    if len(prices) < period + 1:
        return []

    np_prices = np.array(prices)
    # Calculate price changes (deltas)
    deltas = np_prices[1:] - np_prices[:-1]

    # Separate gains and losses
    gains = np.maximum(0, deltas)
    losses = np.maximum(0, -deltas)

    avg_gains = np.zeros_like(gains)
    avg_losses = np.zeros_like(losses)

    # Initial average gain/loss over the first 'period' deltas
    # These correspond to prices from index 0 to 'period'
    avg_gains[period - 1] = np.mean(gains[:period])
    avg_losses[period - 1] = np.mean(losses[:period])

    # Calculate smoothed averages for subsequent periods
    for i in range(period, len(gains)):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period

    # Calculate Relative Strength (RS)
    # Handle division by zero for avg_losses using np.full_like and where clause
    rs = np.divide(avg_gains[period - 1:], avg_losses[period - 1:],
                   out=np.full_like(avg_gains[period - 1:], np.inf),
                   where=avg_losses[period - 1:] != 0)

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi.tolist()


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on the Simple RSI Mean-Reversion Entry rule.

    Initiates a long position when the 14-period RSI crosses below 30 (oversold).
    Initiates a short position when the 14-period RSI crosses above 70 (overbought).
    No trend filter is applied.

    Args:
        data: MarketData object containing tick, warm candle, and cold month data.

    Returns:
        A list of BuySignal or SellSignal objects.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Ensure enough warm candles to calculate at least two RSI values for cross detection
        if len(warm_candles) < MIN_CANDLES_FOR_RSI:
            continue

        # Extract close prices from the warm candles
        close_prices = [candle.close for candle in warm_candles]

        # Calculate RSI values
        rsi_values = _calculate_rsi(close_prices, RSI_PERIOD)

        # Need at least two RSI values to detect a cross (current and previous)
        if len(rsi_values) < 2:
            continue

        # Get the current and previous RSI values
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]

        # The signal timestamp and price correspond to the most recent candle
        latest_candle = warm_candles[-1]
        timestamp = latest_candle.hour  # Candle hour represents the end of the period
        price = latest_candle.close
        
        rule_id = "RSI_MeanReversion_Relaxed_V1"

        # Long signal: RSI crosses below oversold threshold (30)
        # Previous RSI was >= 30 and current RSI is < 30
        if previous_rsi >= RSI_OVERSOLD_THRESHOLD and current_rsi < RSI_OVERSOLD_THRESHOLD:
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=rule_id
            ))
        # Short signal: RSI crosses above overbought threshold (70)
        # Previous RSI was <= 70 and current RSI is > 70
        elif previous_rsi <= RSI_OVERBOUGHT_THRESHOLD and current_rsi > RSI_OVERBOUGHT_THRESHOLD:
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=rule_id
            ))

    return signals