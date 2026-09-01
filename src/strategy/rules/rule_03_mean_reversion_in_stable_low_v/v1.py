from __future__ import annotations
import statistics
import math
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

# Constants for indicator periods and data requirements
MIN_WARM_CANDLES = 20  # Required for Bollinger Bands (20-period)
MIN_HOT_TICKS = 1      # Required for current_tick's last_price and spread_rel
EMA_PERIOD = 9
SMA_PERIOD = 9
RSI_PERIOD = 14
BB_PERIOD = 20

# Rule ID
RULE_ID = "stable_bounce_mean_reversion"

# Helper functions for indicator calculations

def calculate_sma(data: list[float], period: int) -> float | None:
    """Calculates the Simple Moving Average."""
    if len(data) < period:
        return None
    return statistics.mean(data[-period:])

def calculate_ema(data: list[float], period: int) -> float | None:
    """Calculates the Exponential Moving Average."""
    if len(data) < period:
        return None

    # Calculate initial SMA for the first EMA
    sma = calculate_sma(data[:period], period)
    if sma is None:
        return None

    ema = sma
    multiplier = 2 / (period + 1)

    # Apply EMA formula to the rest of the data
    # The current EMA is based on the last value in 'data'
    for price in data[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_rsi(data: list[float], period: int) -> float | None:
    """Calculates the Relative Strength Index."""
    if len(data) <= period: # Need at least period + 1 candles for first gain/loss
        return None

    gains = []
    losses = []

    # Calculate initial gains/losses for the first 'period' changes
    for i in range(1, len(data)):
        change = data[i] - data[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))
    
    if len(gains) < period: # Not enough data for initial average
        return None

    # Calculate initial average gain and loss over the first 'period' changes
    avg_gain = statistics.mean(gains[:period])
    avg_loss = statistics.mean(losses[:period])

    # Smooth subsequent averages using the Wilder's smoothing method (EMA-like)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        # Avoid division by zero. If no losses, RSI is 100 (if there are gains).
        # If no gains and no losses, RSI is 50.
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_band_width(data: list[float], period: int, num_std_dev: float = 2.0) -> float | None:
    """Calculates the Bollinger Band Width as a percentage of the middle band."""
    if len(data) < period:
        return None

    closes = data[-period:]
    sma = statistics.mean(closes)
    
    if sma == 0: # Avoid division by zero if SMA is zero
        return None

    # Calculate standard deviation
    sum_sq_diff = sum([(x - sma) ** 2 for x in closes])
    std_dev = math.sqrt(sum_sq_diff / period)

    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)

    bb_width = ((upper_band - lower_band) / sma) * 100 # As a percentage
    return bb_width

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Mean Reversion in Stable, Low-Volatility Environments' trading rule.

    This rule identifies long entry opportunities for assets exhibiting mean-reversion
    characteristics within stable, low-volatility, and high-liquidity market conditions.
    It triggers a BuySignal when the asset's price is slightly below its short-term
    moving averages (EMA(9), SMA(9)), accompanied by a moderate-to-slightly-oversold
    RSI(14), very low bid-ask spread, and constrained Bollinger Band Width (20-period).
    A SellSignal is issued if the price recovers above the moving averages, RSI becomes
    overbought, or volatility significantly increases.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Data availability checks
        if not pair_data.hot or len(pair_data.hot) < MIN_HOT_TICKS:
            continue
        if not pair_data.warm or len(pair_data.warm) < MIN_WARM_CANDLES:
            continue

        current_tick = pair_data.hot[-1]
        hourly_closes = [c.close for c in pair_data.warm]

        # Ensure enough data for all indicators (BB_PERIOD is the largest lookback)
        if len(hourly_closes) < BB_PERIOD:
            continue

        last_price = current_tick.last_price
        tick_bid_ask_spread_percentage = current_tick.spread_rel

        # Calculate indicators
        hourly_ema_9 = calculate_ema(hourly_closes, EMA_PERIOD)
        hourly_sma_9 = calculate_sma(hourly_closes, SMA_PERIOD)
        hourly_rsi = calculate_rsi(hourly_closes, RSI_PERIOD)
        hourly_bb_width = calculate_bollinger_band_width(hourly_closes, BB_PERIOD)

        # Check if all indicators could be calculated successfully
        if any(x is None for x in [hourly_ema_9, hourly_sma_9, hourly_rsi, hourly_bb_width]):
            continue

        # Buy Signal Conditions:
        # Price slightly below MAs, moderate/slightly oversold RSI, low spread, low BBW.
        if (
            tick_bid_ask_spread_percentage < 0.05 and
            hourly_bb_width < 3.5 and # BBW as percentage, e.g., 3.5%
            hourly_rsi >= 35 and hourly_rsi <= 45 and
            last_price < hourly_ema_9 and
            last_price < hourly_sma_9 and
            last_price > hourly_ema_9 * 0.992 and # Price is within 0.8% below EMA
            last_price > hourly_sma_9 * 0.992    # Price is within 0.8% below SMA
        ):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=last_price,
                rule_id=RULE_ID,
                confidence=0.7 # Moderate confidence for entry
            ))

        # Sell Signal Conditions (to close an existing long position):
        # Price recovers above MAs, RSI becomes overbought, or volatility increases significantly.
        # Using `elif` ensures that if a BuySignal condition is met, a SellSignal is not
        # simultaneously issued for the same pair.
        elif (
            last_price > hourly_ema_9 or
            last_price > hourly_sma_9 or
            hourly_rsi > 60 or
            hourly_bb_width > 5.0
        ):
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=last_price,
                rule_id=RULE_ID,
                confidence=0.8 # Higher confidence for exit when conditions are met
            ))

    return signals