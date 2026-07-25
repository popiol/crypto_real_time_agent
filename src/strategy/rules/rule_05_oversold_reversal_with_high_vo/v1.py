from __future__ import annotations
import math
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick


def calculate_rsi(closes: list[float], period: int) -> list[float]:
    """
    Calculates the Relative Strength Index (RSI) for a given list of closing prices.
    Returns a list of RSI values, with the latest RSI at the end.
    """
    if len(closes) < period + 1:
        return []

    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]

    avg_gains: list[float] = []
    avg_losses: list[float] = []

    # Initial average for the first 'period' changes
    initial_gains_sum = sum(gains[:period])
    initial_losses_sum = sum(losses[:period])
    avg_gains.append(initial_gains_sum / period)
    avg_losses.append(initial_losses_sum / period)

    # Subsequent EMA calculation
    for i in range(period, len(gains)):
        prev_avg_gain = avg_gains[-1]
        prev_avg_loss = avg_losses[-1]
        current_gain = gains[i]
        current_loss = losses[i]

        new_avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
        new_avg_loss = (prev_avg_loss * (period - 1) + current_loss) / period
        avg_gains.append(new_avg_gain)
        avg_losses.append(new_avg_loss)

    rsi_values: list[float] = []
    for i in range(len(avg_gains)):
        if avg_losses[i] == 0:
            rs = float('inf')  # Treat as very strong upward momentum if no losses
        else:
            rs = avg_gains[i] / avg_losses[i]
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

    return rsi_values


def calculate_sma(prices: list[float], period: int) -> float | None:
    """
    Calculates the Simple Moving Average (SMA) for the last 'period' prices.
    """
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
    """
    Calculates the Average True Range (ATR) using an EMA-like smoothing.
    Returns the latest ATR value.
    """
    if len(highs) < period + 1:  # Need period + 1 candles to calculate 'period' true ranges
        return None

    true_ranges: list[float] = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_prev_close = abs(highs[i] - closes[i - 1])
        low_prev_close = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(high_low, high_prev_close, low_prev_close))

    if len(true_ranges) < period:
        # This case should ideally be caught by the initial len(highs) check,
        # but serves as an additional safeguard.
        return None

    atrs: list[float] = []
    # Initial ATR is simple average of first 'period' true ranges
    atrs.append(sum(true_ranges[:period]) / period)

    # Subsequent EMA calculation
    for i in range(period, len(true_ranges)):
        prev_atr = atrs[-1]
        current_tr = true_ranges[i]
        new_atr = (prev_atr * (period - 1) + current_tr) / period
        atrs.append(new_atr)

    return atrs[-1]


def calculate_volume_anomaly(current_volume: float, volumes: list[float], period: int) -> float | None:
    """
    Calculates the volume anomaly as current volume divided by the average volume
    over the last 'period' candles. Returns 0.0 if average volume is zero.
    """
    if len(volumes) < period:
        return None

    avg_volume = sum(volumes[-period:]) / period

    if avg_volume == 0:
        # Per pseudocode, return 0.0 if average volume is zero.
        # This will prevent a signal as the condition is > 1.0.
        return 0.0

    return current_volume / avg_volume


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates BUY signals based on an 'Oversold Reversal with High Volume Confirmation' strategy.
    
    The strategy triggers a BUY signal when:
    - Hourly Relative Strength Index (RSI) is between 45 and 57.
    - Hourly Keltner Channel Position is between 0.1 and 0.4 (relative to the channel's centerline).
    - Hourly Volume Anomaly is greater than 1.0.
    """
    signals: list[BuySignal | SellSignal] = []

    # Define indicator periods and multipliers
    RSI_PERIOD = 14
    KC_SMA_PERIOD = 20
    KC_ATR_PERIOD = 20
    KC_ATR_MULTIPLIER = 2.0
    VOL_ANOMALY_PERIOD = 20

    # Minimum candles required for all indicators:
    # RSI(14) needs 14+1 = 15 candles
    # SMA(20) needs 20 candles
    # ATR(20) needs 20+1 = 21 candles
    # Volume Anomaly (20) needs 20 candles
    MIN_CANDLES = max(RSI_PERIOD + 1, KC_SMA_PERIOD, KC_ATR_PERIOD + 1, VOL_ANOMALY_PERIOD)

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure sufficient warm data for indicator calculations
        if len(warm_candles) < MIN_CANDLES:
            continue

        # Ensure hot data is available for the latest price for signal generation
        if not hot_ticks:
            continue

        latest_candle = warm_candles[-1]
        current_close = latest_candle.close
        current_volume = latest_candle.volume

        # Extract data series for indicator calculations
        closes = [c.close for c in warm_candles]
        highs = [c.high for c in warm_candles]
        lows = [c.low for c in warm_candles]
        volumes = [c.volume for c in warm_candles]

        # 1. Calculate 14-period RSI
        rsi_values = calculate_rsi(closes, RSI_PERIOD)
        if not rsi_values:
            continue  # Should not happen if MIN_CANDLES check passes, but for robustness
        current_rsi = rsi_values[-1]

        # 2. Calculate Keltner Channel Position
        sma_20 = calculate_sma(closes, KC_SMA_PERIOD)
        atr_20 = calculate_atr(highs, lows, closes, KC_ATR_PERIOD)

        keltner_position: float = 0.0
        if sma_20 is not None and atr_20 is not None and atr_20 > 0:
            keltner_position = (current_close - sma_20) / (KC_ATR_MULTIPLIER * atr_20)
        else:
            # If SMA, ATR, or ATR is zero, Keltner position cannot be reliably calculated.
            # Skip this pair as the condition cannot be met.
            continue

        # 3. Calculate 20-period Volume Anomaly
        volume_anomaly = calculate_volume_anomaly(current_volume, volumes, VOL_ANOMALY_PERIOD)
        if volume_anomaly is None:
            continue  # Should not happen if MIN_CANDLES check passes

        # BUY Condition Check:
        # RSI between 45 and 57
        # Keltner Channel Position between 0.1 and 0.4
        # Volume Anomaly greater than 1.0
        if (45 <= current_rsi <= 57 and
                0.1 <= keltner_position <= 0.4 and
                volume_anomaly > 1.0):

            signals.append(BuySignal(
                pair=pair,
                timestamp=hot_ticks[-1].polled_at,
                price=hot_ticks[-1].last_price,
                rule_id="oversold_reversal_high_vol_conf",
                confidence=1.0  # Assign a default confidence
            ))

    return signals