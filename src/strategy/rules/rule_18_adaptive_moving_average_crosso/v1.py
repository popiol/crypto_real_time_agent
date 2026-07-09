from __future__ import annotations
import statistics
from datetime import datetime
import numpy as np

from src.agent.models import BuySignal, MarketData, SellSignal

# --- Rule Parameters ---
# KAMA parameters for the fast moving average
AMA_FAST_ER_PERIOD = 10  # Efficiency Ratio period for fast KAMA
AMA_FAST_EMA_PERIOD_FAST = 2  # Fast EMA smoothing period for fast KAMA
AMA_FAST_EMA_PERIOD_SLOW = 30 # Slow EMA smoothing period for fast KAMA

# KAMA parameters for the slow moving average
AMA_SLOW_ER_PERIOD = 20  # Efficiency Ratio period for slow KAMA
AMA_SLOW_EMA_PERIOD_FAST = 2  # Fast EMA smoothing period for slow KAMA
AMA_SLOW_EMA_PERIOD_SLOW = 30 # Slow EMA smoothing period for slow KAMA

# Volume confirmation parameters
VOLUME_SMA_PERIOD = 60  # Period for Simple Moving Average of 24h volume from ticks
VOLUME_CONFIRMATION_FACTOR = 1.2  # Current volume must be > SMA * factor

# Minimum data requirements
MIN_CANDLES_FOR_AMA = max(AMA_FAST_ER_PERIOD, AMA_SLOW_ER_PERIOD) + 1
MIN_TICKS_FOR_VOLUME_SMA = VOLUME_SMA_PERIOD

# --- Helper Function for KAMA Calculation ---
def _calculate_kama(
    prices: list[float],
    er_period: int,
    fast_ema_period: int,
    slow_ema_period: int
) -> np.ndarray:
    prices_arr = np.array(prices, dtype=float)
    n = len(prices_arr)

    if n < er_period:
        return np.array([])

    kama_values = np.zeros(n)
    fast_alpha = 2 / (fast_ema_period + 1)
    slow_alpha = 2 / (slow_ema_period + 1)

    kama_values[er_period - 1] = prices_arr[er_period - 1]

    for i in range(er_period, n):
        change = abs(prices_arr[i] - prices_arr[i - er_period])
        
        volatility_window = prices_arr[i - er_period : i + 1]
        volatility = np.sum(np.abs(np.diff(volatility_window)))

        if volatility == 0:
            er = 0.0
        else:
            er = change / volatility

        sc = (er * (fast_alpha - slow_alpha) + slow_alpha)**2

        kama_values[i] = kama_values[i-1] + sc * (prices_arr[i] - kama_values[i-1])

    return kama_values[er_period - 1:]

# --- Main Signal Function ---
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        candles = pair_data.warm
        ticks = pair_data.hot

        if len(candles) < MIN_CANDLES_FOR_AMA:
            continue
        if len(ticks) < MIN_TICKS_FOR_VOLUME_SMA:
            continue
        
        close_prices = [c.close for c in candles]
        volumes_24h = [t.volume_24h for t in ticks]

        ama_fast_values = _calculate_kama(
            close_prices,
            AMA_FAST_ER_PERIOD,
            AMA_FAST_EMA_PERIOD_FAST,
            AMA_FAST_EMA_PERIOD_SLOW
        )
        ama_slow_values = _calculate_kama(
            close_prices,
            AMA_SLOW_ER_PERIOD,
            AMA_SLOW_EMA_PERIOD_FAST,
            AMA_SLOW_EMA_PERIOD_SLOW
        )
        
        if len(ama_fast_values) < 2 or len(ama_slow_values) < 2:
            continue

        current_ama_fast = ama_fast_values[-1]
        prev_ama_fast = ama_fast_values[-2]
        current_ama_slow = ama_slow_values[-1]
        prev_ama_slow = ama_slow_values[-2]

        current_volume = volumes_24h[-1]
        volume_sma = statistics.mean(volumes_24h[-VOLUME_SMA_PERIOD:])
        
        volume_confirmed = current_volume > (volume_sma * VOLUME_CONFIRMATION_FACTOR)

        last_tick = ticks[-1]
        signal_timestamp = last_tick.polled_at
        signal_price = last_tick.last_price

        if (prev_ama_fast <= prev_ama_slow and
            current_ama_fast > current_ama_slow and
            volume_confirmed):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
                rule_id="adaptive_ama_volume_crossover"
            ))

        elif (prev_ama_fast >= prev_ama_slow and
              current_ama_fast < current_ama_slow and
              volume_confirmed):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=signal_timestamp,
                price=signal_price,
                rule_id="adaptive_ama_volume_crossover"
            ))

    return signals