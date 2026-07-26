from __future__ import annotations
import statistics
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, SellSignal, MarketData, WarmCandle

RSI_PERIOD = 14
KCP_SMA_PERIOD = 20
KCP_ATR_PERIOD = 10
KCP_ATR_MULTIPLIER = 2.0
BBW_PERIOD = 20
BBW_STDDEV_MULTIPLIER = 2.0

MIN_CANDLES_REQUIRED = 20

RSI_LOWER_BOUND = 45
RSI_UPPER_BOUND = 60
KCP_LOWER_BOUND = 20
KCP_UPPER_BOUND = 70
BBW_UPPER_BOUND = 7.0

TAKE_PROFIT_FACTOR = 1.03
STOP_LOSS_FACTOR = 0.98

def _calculate_sma(data: list[float], period: int) -> float:
    if len(data) < period:
        return np.nan
    return statistics.mean(data[-period:])

def _calculate_stddev(data: list[float], period: int) -> float:
    if len(data) < period:
        return np.nan
    return np.std(data[-period:])

def _calculate_true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))

def _calculate_atr(warm_candles: list[WarmCandle], period: int) -> float:
    if len(warm_candles) < period + 1:
        return np.nan

    true_ranges = []
    for i in range(1, len(warm_candles)):
        current_candle = warm_candles[i]
        prev_candle = warm_candles[i-1]
        tr = _calculate_true_range(current_candle.high, current_candle.low, prev_candle.close)
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return np.nan
    
    return statistics.mean(true_ranges[-period:])

def _calculate_rsi(closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        return np.nan

    gains_full = [0.0] * len(closes)
    losses_full = [0.0] * len(closes)

    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains_full[i] = change
        else:
            losses_full[i] = abs(change)

    avg_gain_list = [np.nan] * len(closes)
    avg_loss_list = [np.nan] * len(closes)

    if len(closes) > period:
        avg_gain_list[period] = statistics.mean(gains_full[1:period+1])
        avg_loss_list[period] = statistics.mean(losses_full[1:period+1])

    for i in range(period + 1, len(closes)):
        avg_gain_list[i] = ((avg_gain_list[i-1] * (period - 1)) + gains_full[i]) / period
        avg_loss_list[i] = ((avg_loss_list[i-1] * (period - 1)) + losses_full[i]) / period

    current_avg_gain = avg_gain_list[-1]
    current_avg_loss = avg_loss_list[-1]

    if np.isnan(current_avg_gain) or np.isnan(current_avg_loss):
        return np.nan

    if current_avg_loss == 0:
        return 100.0
    if current_avg_gain == 0:
        return 0.0

    rs = current_avg_gain / current_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_keltner_channel_percentage(
    warm_candles: list[WarmCandle], sma_period: int, atr_period: int, atr_multiplier: float
) -> float:
    closes = [c.close for c in warm_candles]
    current_close = closes[-1]

    middle_band = _calculate_sma(closes, sma_period)
    if np.isnan(middle_band):
        return np.nan

    atr_value = _calculate_atr(warm_candles, atr_period)
    if np.isnan(atr_value):
        return np.nan

    channel_width = atr_value * atr_multiplier
    upper_band = middle_band + channel_width
    lower_band = middle_band - channel_width

    if (upper_band - lower_band) == 0:
        return 50.0
    
    kcp = ((current_close - lower_band) / (upper_band - lower_band)) * 100
    return kcp

def _calculate_bollinger_bands_width(
    closes: list[float], period: int, stddev_multiplier: float
) -> float:
    if len(closes) < period:
        return np.nan

    middle_band = _calculate_sma(closes, period)
    if np.isnan(middle_band):
        return np.nan

    std_dev = _calculate_stddev(closes, period)
    if np.isnan(std_dev):
        return np.nan

    upper_band = middle_band + (std_dev * stddev_multiplier)
    lower_band = middle_band - (std_dev * stddev_multiplier)

    if middle_band == 0:
        return np.nan
    
    bb_width_percentage = ((upper_band - lower_band) / middle_band) * 100
    return bb_width_percentage

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue

        closes = [c.close for c in warm_candles]
        current_close = closes[-1]
        
        current_rsi = _calculate_rsi(closes, period=RSI_PERIOD)
        if np.isnan(current_rsi):
            continue

        keltner_channel_percentage = _calculate_keltner_channel_percentage(
            warm_candles, sma_period=KCP_SMA_PERIOD, atr_period=KCP_ATR_PERIOD, atr_multiplier=KCP_ATR_MULTIPLIER
        )
        if np.isnan(keltner_channel_percentage):
            continue

        bb_width = _calculate_bollinger_bands_width(
            closes, period=BBW_PERIOD, stddev_multiplier=BBW_STDDEV_MULTIPLIER
        )
        if np.isnan(bb_width):
            continue

        rsi_condition = RSI_LOWER_BOUND <= current_rsi <= RSI_UPPER_BOUND
        kcp_condition = KCP_LOWER_BOUND <= keltner_channel_percentage <= KCP_UPPER_BOUND
        bbw_condition = bb_width < BBW_UPPER_BOUND

        if rsi_condition and kcp_condition and bbw_condition:
            signals.append(
                BuySignal(
                    pair=pair,
                    timestamp=warm_candles[-1].hour,
                    price=current_close,
                    rule_id="moderate_uptrend_vol_filter_entry",
                    indicators={
                        "rsi": current_rsi,
                        "kcp": keltner_channel_percentage,
                        "bbw": bb_width,
                    },
                    take_profit_target=current_close * TAKE_PROFIT_FACTOR,
                    stop_loss_level=current_close * STOP_LOSS_FACTOR
                )
            )
    return signals