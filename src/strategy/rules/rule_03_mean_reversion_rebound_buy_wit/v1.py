from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
import numpy as np

# Rule-specific constants
RSI_PERIOD = 14
KC_EMA_PERIOD = 20
KC_ATR_PERIOD = 10
KC_ATR_MULTIPLIER = 2.0
VOLUME_ANOMALY_PERIOD = 10

# RSI thresholds
RSI_MIN_BUY = 35
RSI_MAX_BUY = 45
RSI_DEEPLY_OVERSOLD = 30

# Keltner Channel Position thresholds
KCP_MIN_BUY = -0.5
KCP_MAX_BUY = 0.1
KCP_DEEPLY_OVERSOLD = -0.5

# Volume Anomaly threshold
VOLUME_ANOMALY_THRESHOLD = 1.0

# Minimum candles required for each indicator
# RSI needs `period + 1` candles for the first RSI value (e.g., 14 diffs for period 14)
MIN_CANDLES_RSI = RSI_PERIOD + 1
# EMA needs `period` candles to calculate the first EMA value (using SMA for initialization)
MIN_CANDLES_KC_EMA = KC_EMA_PERIOD
# ATR needs `period + 1` candles because TR calculation requires previous close
MIN_CANDLES_KC_ATR = KC_ATR_PERIOD + 1
# Volume Anomaly needs `lookback_period + 1` candles (N candles for average + 1 for current)
MIN_CANDLES_VOLUME_ANOMALY = VOLUME_ANOMALY_PERIOD + 1

# Overall minimum warm candles required for all indicators to be computable
MIN_WARM_CANDLES = max(
    MIN_CANDLES_RSI,
    MIN_CANDLES_KC_EMA,
    MIN_CANDLES_KC_ATR,
    MIN_CANDLES_VOLUME_ANOMALY
)


def _calculate_rsi(candles: list[WarmCandle], period: int) -> float | None:
    """Calculates the Relative Strength Index (RSI) for the last candle."""
    if len(candles) < period + 1:
        return None

    closes = np.array([c.close for c in candles])
    price_diffs = np.diff(closes)

    gains = np.where(price_diffs > 0, price_diffs, 0)
    losses = np.where(price_diffs < 0, np.abs(price_diffs), 0)

    # Calculate initial average gain/loss using SMA over the first 'period' differences
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Apply Wilder's smoothing for subsequent values up to the latest
    for i in range(period, len(price_diffs)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0  # Handle division by zero
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_ema(prices: np.ndarray, period: int) -> float | None:
    """Calculates the Exponential Moving Average (EMA) for the last price."""
    if len(prices) < period:
        return None
    
    alpha = 2 / (period + 1)
    # Initialize EMA with SMA of the first 'period' prices
    ema = np.mean(prices[:period])
    
    # Calculate EMA iteratively for subsequent prices
    for i in range(period, len(prices)):
        ema = alpha * prices[i] + (1 - alpha) * ema
        
    return ema


def _calculate_atr(candles: list[WarmCandle], period: int) -> float | None:
    """Calculates the Average True Range (ATR) for the last candle."""
    if len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i-1].close),
            abs(candles[i].low - candles[i-1].close)
        )
        true_ranges.append(tr)
    
    if len(true_ranges) < period: # Should not happen if initial len(candles) check passes
        return None
    
    # ATR is typically a simple moving average of the True Ranges
    atr = np.mean(true_ranges[-period:])
    return atr


def _get_keltner_channel_position(
    candles: list[WarmCandle],
    ema_period: int,
    atr_period: int,
    atr_multiplier: float,
    current_price: float
) -> float | None:
    """Calculates the Keltner Channel Position."""
    min_candles_required = max(ema_period, atr_period + 1)
    if len(candles) < min_candles_required:
        return None

    closes = np.array([c.close for c in candles])
    
    center_line = _calculate_ema(closes, ema_period)
    if center_line is None:
        return None
    
    atr = _calculate_atr(candles, atr_period)
    if atr is None:
        return None

    if atr * atr_multiplier == 0:
        return 0.0 # Avoid division by zero, treat as flat channel
    
    return (current_price - center_line) / (atr * atr_multiplier)


def _get_volume_anomaly(candles: list[WarmCandle], lookback_period: int) -> float | None:
    """Calculates the Volume Anomaly."""
    if len(candles) < lookback_period + 1:
        return None
    
    volumes = np.array([c.volume for c in candles])
    
    current_volume = volumes[-1]
    
    # Calculate average volume over the lookback_period *before* the current candle
    average_volume = np.mean(volumes[-(lookback_period + 1):-1])

    if average_volume == 0:
        return 0.0 if current_volume == 0 else float('inf') # Handle division by zero
    
    return current_volume / average_volume


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Mean Reversion Rebound Buy with Volume Confirmation' rule.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        hot_ticks = pair_data.hot
        warm_candles = pair_data.warm

        if not hot_ticks:
            continue
        
        # Ensure enough warm candles for all indicators
        if len(warm_candles) < MIN_WARM_CANDLES:
            continue

        current_price = hot_ticks[-1].last_price
        current_timestamp = hot_ticks[-1].polled_at

        # Calculate indicators
        rsi_val = _calculate_rsi(warm_candles, RSI_PERIOD)
        if rsi_val is None:
            continue

        kcp_val = _get_keltner_channel_position(
            warm_candles, KC_EMA_PERIOD, KC_ATR_PERIOD, KC_ATR_MULTIPLIER, current_price
        )
        if kcp_val is None:
            continue

        volume_anomaly_val = _get_volume_anomaly(warm_candles, VOLUME_ANOMALY_PERIOD)
        if volume_anomaly_val is None:
            continue

        # Apply the rule logic
        # IF (RSI >= 35 AND RSI <= 45 AND KCP >= -0.5 AND KCP <= 0.1 AND VA > 1.0
        # AND NOT (RSI < 30 AND KCP < -0.5)) THEN SIGNAL: BUY
        
        condition_rsi_range = (rsi_val >= RSI_MIN_BUY and rsi_val <= RSI_MAX_BUY)
        condition_kcp_range = (kcp_val >= KCP_MIN_BUY and kcp_val <= KCP_MAX_BUY)
        condition_volume_confirmation = (volume_anomaly_val > VOLUME_ANOMALY_THRESHOLD)
        
        # Explicitly avoid deeply oversold conditions
        condition_avoid_deeply_oversold = not (
            rsi_val < RSI_DEEPLY_OVERSOLD and kcp_val < KCP_DEEPLY_OVERSOLD
        )

        if (
            condition_rsi_range and
            condition_kcp_range and
            condition_volume_confirmation and
            condition_avoid_deeply_oversold
        ):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_timestamp,
                price=current_price,
                rule_id="mean_reversion_rebound_buy_v1",
                confidence=None # Confidence is not specified in the rule idea
            ))

    return signals