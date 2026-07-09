from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0
MIN_CANDLES = max(SLOW_PERIOD + SIGNAL_PERIOD, 3 * ADX_PERIOD) + 5

def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    if len(prices) < period:
        return np.array([])

    ema = np.zeros_like(prices, dtype=float)
    
    ema[period - 1] = np.mean(prices[:period])
    
    alpha = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    
    return ema[period - 1:]

def _calculate_macd(
    prices: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if len(prices) < slow_period:
        return np.array([]), np.array([]), np.array([])

    ema_fast = _calculate_ema(prices, fast_period)
    ema_slow = _calculate_ema(prices, slow_period)

    if not ema_fast.size or not ema_slow.size:
        return np.array([]), np.array([]), np.array([])

    if len(ema_fast) > len(ema_slow):
        ema_fast = ema_fast[len(ema_fast) - len(ema_slow):]
    elif len(ema_slow) > len(ema_fast):
        ema_slow = ema_slow[len(ema_slow) - len(ema_fast):]
        
    macd_line = ema_fast - ema_slow

    if len(macd_line) < signal_period:
        return np.array([]), np.array([]), np.array([])
        
    signal_line = _calculate_ema(macd_line, signal_period)

    if not signal_line.size:
        return np.array([]), np.array([]), np.array([])

    macd_line = macd_line[len(macd_line) - len(signal_line):]
    
    macd_histogram = macd_line - signal_line

    return macd_line, signal_line, macd_histogram

def _calculate_adx(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    period: int
) -> np.ndarray:
    if len(high_prices) < period * 2 or len(low_prices) < period * 2 or len(close_prices) < period * 2:
        return np.array([])

    tr = np.zeros_like(high_prices, dtype=float)
    plus_dm = np.zeros_like(high_prices, dtype=float)
    minus_dm = np.zeros_like(high_prices, dtype=float)

    for i in range(1, len(high_prices)):
        tr[i] = max(
            high_prices[i] - low_prices[i],
            abs(high_prices[i] - close_prices[i-1]),
            abs(low_prices[i] - close_prices[i-1])
        )

        up_move = high_prices[i] - high_prices[i-1]
        down_move = low_prices[i-1] - low_prices[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0

    smoothed_tr = np.zeros_like(tr, dtype=float)
    smoothed_plus_dm = np.zeros_like(plus_dm, dtype=float)
    smoothed_minus_dm = np.zeros_like(minus_dm, dtype=float)

    smoothed_tr[period] = np.sum(tr[1:period+1])
    smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])

    for i in range(period + 1, len(tr)):
        smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / period) + tr[i]
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / period) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / period) + minus_dm[i]

    di_plus = np.zeros_like(smoothed_plus_dm, dtype=float)
    di_minus = np.zeros_like(smoothed_minus_dm, dtype=float)

    valid_indices = smoothed_tr > 0
    di_plus[valid_indices] = (smoothed_plus_dm[valid_indices] / smoothed_tr[valid_indices]) * 100
    di_minus[valid_indices] = (smoothed_minus_dm[valid_indices] / smoothed_tr[valid_indices]) * 100

    dx = np.zeros_like(di_plus, dtype=float)
    sum_di = di_plus + di_minus
    valid_dx_indices = sum_di > 0
    dx[valid_dx_indices] = (np.abs(di_plus[valid_dx_indices] - di_minus[valid_dx_indices]) / sum_di[valid_dx_indices]) * 100

    adx = np.zeros_like(dx, dtype=float)
    
    if len(dx) < 2 * period:
        return np.array([])
        
    adx[2 * period - 1] = np.sum(dx[period:2*period]) / period

    for i in range(2 * period, len(dx)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx[2 * period - 1:]

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        
        if len(warm_candles) < MIN_CANDLES:
            continue

        close_prices = np.array([c.close for c in warm_candles])
        high_prices = np.array([c.high for c in warm_candles])
        low_prices = np.array([c.low for c in warm_candles])
        timestamps = [c.hour for c in warm_candles]

        macd_line, signal_line, _ = _calculate_macd(
            close_prices, FAST_PERIOD, SLOW_PERIOD, SIGNAL_PERIOD
        )
        
        if not macd_line.size or not signal_line.size:
            continue

        adx = _calculate_adx(high_prices, low_prices, close_prices, ADX_PERIOD)
        
        if not adx.size:
            continue

        current_macd = macd_line[-1]
        prev_macd = macd_line[-2] if len(macd_line) >= 2 else None
        
        current_signal = signal_line[-1]
        prev_signal = signal_line[-2] if len(signal_line) >= 2 else None

        current_adx = adx[-1]

        if prev_macd is None or prev_signal is None:
            continue

        last_candle_timestamp = timestamps[-1]
        last_candle_close = close_prices[-1]

        if (prev_macd < prev_signal and current_macd > current_signal) and (current_adx > ADX_THRESHOLD):
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=last_candle_timestamp,
                    price=last_candle_close,
                    rule_id="01c21bea-82a8-4ce4-8e22-76c0dd1b2e47"
                )
            )
        elif (prev_macd > prev_signal and current_macd < current_signal) and (current_adx > ADX_THRESHOLD):
            signals.append(
                SellSignal(
                    pair=pair,
                    timestamp=last_candle_timestamp,
                    price=last_candle_close,
                    rule_id="01c21bea-82a8-4ce4-8e22-76c0dd1b2e47"
                )
            )

    return signals