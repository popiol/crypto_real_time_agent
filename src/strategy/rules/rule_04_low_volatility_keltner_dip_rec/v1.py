from __future__ import annotations
import numpy as np
import statistics
from datetime import datetime
from src.agent.models import BuySignal, SellSignal, MarketData, Tick, WarmCandle, ColdMonth

# Helper functions for indicator calculations
def _sma(data: np.ndarray, period: int) -> float | None:
    """Calculates the Simple Moving Average for the last `period` values."""
    if len(data) < period:
        return None
    return np.mean(data[-period:])

def _std_dev(data: np.ndarray, period: int) -> float | None:
    """Calculates the Standard Deviation for the last `period` values."""
    if len(data) < period:
        return None
    return np.std(data[-period:])

def _ema(data: np.ndarray, period: int) -> float | None:
    """Calculates the Exponential Moving Average for the last value in the data series."""
    if len(data) < period: # EMA requires at least 'period' data points for a meaningful calculation
        return None
    
    alpha = 2 / (period + 1)
    ema_values = np.zeros_like(data)
    
    # Initialize the first EMA with the first data point, or SMA of initial period
    # For simplicity and consistency for the last value, we calculate the series.
    ema_values[0] = data[0]

    for i in range(1, len(data)):
        ema_values[i] = (data[i] * alpha) + (ema_values[i-1] * (1 - alpha))
    
    # Return the last EMA value only if sufficient data points were processed
    return ema_values[-1]


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float | None:
    """Calculates the Average True Range."""
    # Need at least period + 1 candles to compute 'period' True Ranges.
    if len(highs) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        true_ranges.append(tr)
    
    # Calculate EMA of True Ranges
    if not true_ranges:
        return None
    
    true_ranges_np = np.array(true_ranges)
    # Ensure there are enough true_ranges for the EMA calculation
    if len(true_ranges_np) < period:
        return None

    return _ema(true_ranges_np, period)


# Indicator calculation functions
def calculate_bollinger_band_width(closes: np.ndarray, period: int = 20, num_std_dev: int = 2) -> float | None:
    """Calculates the Bollinger Band Width."""
    if len(closes) < period:
        return None
    
    middle_band = _sma(closes, period)
    std = _std_dev(closes, period)

    if middle_band is None or std is None or middle_band == 0:
        return None
    
    upper_band = middle_band + (num_std_dev * std)
    lower_band = middle_band - (num_std_dev * std)
    
    if middle_band == 0: # Avoid division by zero
        return None

    return (upper_band - lower_band) / middle_band


def calculate_rsi(closes: np.ndarray, period: int = 14) -> float | None:
    """Calculates the Relative Strength Index (RSI)."""
    # Need at least period+1 closes for initial price differences (deltas)
    if len(closes) < period + 1:
        return None

    deltas = np.diff(closes) # len(deltas) = len(closes) - 1
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0) # Make losses positive

    # Calculate initial average gain and loss over the first 'period' deltas
    # (which corresponds to period+1 closes)
    initial_avg_gain = _sma(gains[:period], period)
    initial_avg_loss = _sma(losses[:period], period)

    if initial_avg_gain is None or initial_avg_loss is None:
        return None

    # Apply Wilder's smoothing for subsequent periods
    current_avg_gain = initial_avg_gain
    current_avg_loss = initial_avg_loss

    for i in range(period, len(gains)):
        current_avg_gain = ((current_avg_gain * (period - 1)) + gains[i]) / period
        current_avg_loss = ((current_avg_loss * (period - 1)) + losses[i]) / period

    if current_avg_loss == 0:
        # If no losses, RSI is 100 if there are gains, else 50 (neutral)
        return 100.0 if current_avg_gain > 0 else 50.0
    
    rs = current_avg_gain / current_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_keltner_channel_percentage(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    ema_period: int = 20, atr_period: int = 10, atr_multiplier: int = 2
) -> float | None:
    """Calculates the Keltner Channel Percentage."""
    # Ensure enough data for both EMA and ATR calculations.
    # ATR needs atr_period + 1 candles for true ranges and then atr_period for its EMA.
    # EMA needs ema_period candles.
    # So, we need at least max(ema_period, atr_period + 1) candles.
    if len(closes) < max(ema_period, atr_period + 1):
        return None

    middle_band = _ema(closes, ema_period)
    atr_val = _atr(highs, lows, closes, atr_period)

    if middle_band is None or atr_val is None:
        return None
    
    upper_band = middle_band + (atr_multiplier * atr_val)
    lower_band = middle_band - (atr_multiplier * atr_val)

    latest_close = closes[-1]

    if (upper_band - lower_band) == 0: # Avoid division by zero if channel width is zero
        return None

    keltner_percent = ((latest_close - lower_band) / (upper_band - lower_band)) * 100
    return keltner_percent


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Evaluates the 'Low Volatility Keltner Dip/Recovery Buy' trading rule.

    This rule targets buying opportunities in low-volatility, neutral-to-slightly-oversold
    market conditions when the price is either experiencing a significant dip within the
    Keltner Channel or showing an initial recovery from a shallow dip.
    It requires tight bid-ask spreads to minimize transaction costs.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm data for all indicators
        # BBANDS(20) and Keltner EMA(20) require at least 20 candles.
        # RSI(14) requires 14+1 = 15 candles.
        # Keltner ATR(10) requires 10+1 = 11 candles for TRs, then 10 for EMA of TRs, effectively 20 for a stable EMA.
        # Thus, 20 is the minimum number of warm candles needed.
        if len(pair_data.warm) < 20:
            continue

        # Prepare numpy arrays from warm candle data for efficient calculation
        closes = np.array([c.close for c in pair_data.warm])
        highs = np.array([c.high for c in pair_data.warm])
        lows = np.array([c.low for c in pair_data.warm])

        # Calculate hourly indicators
        hourly_bb_width = calculate_bollinger_band_width(closes, period=20)
        hourly_rsi = calculate_rsi(closes, period=14)
        hourly_keltner_percent = calculate_keltner_channel_percentage(closes, highs, lows, ema_period=20, atr_period=10)

        # Get latest tick data for spread calculation
        if not pair_data.hot:
            continue
        latest_tick = pair_data.hot[-1]
        
        # Ensure last_price is not zero to avoid division by zero
        if latest_tick.last_price == 0:
            continue
        tick_spread_percent = (latest_tick.ask_price - latest_tick.bid_price) / latest_tick.last_price * 100

        # Define thresholds based on the rule's rationale
        bb_width_threshold = 0.03
        rsi_lower_bound = 34
        rsi_upper_bound = 48
        spread_percent_threshold = 0.17
        keltner_dip_threshold = 40
        keltner_recovery_lower = 50
        keltner_recovery_upper = 68

        # Check primary market conditions
        is_low_volatility = hourly_bb_width is not None and hourly_bb_width < bb_width_threshold
        is_neutral_oversold_rsi = hourly_rsi is not None and rsi_lower_bound <= hourly_rsi <= rsi_upper_bound
        is_tight_spread = tick_spread_percent < spread_percent_threshold

        # Check Keltner Channel Percentage for specific entry scenarios
        is_deep_keltner_dip = hourly_keltner_percent is not None and hourly_keltner_percent < keltner_dip_threshold
        is_keltner_recovery = hourly_keltner_percent is not None and keltner_recovery_lower <= hourly_keltner_percent <= keltner_recovery_upper

        # Combine all conditions for a BUY signal
        if (
            is_low_volatility and 
            is_neutral_oversold_rsi and 
            is_tight_spread and 
            (is_deep_keltner_dip or is_keltner_recovery)
        ):
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_tick.polled_at,
                price=latest_tick.last_price,
                rule_id='keltner_dip_recovery_buy'
            ))

    return signals