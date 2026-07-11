import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# Parameters
PERIOD_BB = 20
STD_DEV_BB = 2
PERIOD_MFI = 14
OVERSOLD_MFI = 20
OVERBOUGHT_MFI = 80
PERIOD_VOL_AVG = 20
VOL_MULTIPLIER = 1.5
REJECTION_RATIO = 0.7

# Minimum required candles for all calculations
# BB requires PERIOD_BB candles.
# MFI requires PERIOD_MFI + 1 candles (for typical price comparison with previous candle).
# Average Volume requires PERIOD_VOL_AVG candles.
MIN_CANDLES = max(PERIOD_BB, PERIOD_MFI + 1, PERIOD_VOL_AVG)

def _calculate_sma(data: np.ndarray, period: int) -> float:
    """Calculates the Simple Moving Average for the last `period` values."""
    if len(data) < period:
        return np.nan
    return np.mean(data[-period:])

def _calculate_stddev(data: np.ndarray, period: int) -> float:
    """Calculates the Standard Deviation for the last `period` values."""
    if len(data) < period:
        return np.nan
    return np.std(data[-period:])

def _calculate_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> float:
    """Calculates the Money Flow Index."""
    if len(high) <= period: # Need period + 1 for typical price comparison
        return np.nan

    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    positive_money_flow = np.zeros_like(raw_money_flow)
    negative_money_flow = np.zeros_like(raw_money_flow)

    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_money_flow[i] = raw_money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_money_flow[i] = raw_money_flow[i]

    # Sum over the last `period` candles
    pmf_sum = np.sum(positive_money_flow[-period:])
    nmf_sum = np.sum(negative_money_flow[-period:])

    if nmf_sum == 0:
        if pmf_sum == 0:
            return 50.0 # Neutral MFI if no money flow
        return 100.0 # All positive money flow
    
    money_ratio = pmf_sum / nmf_sum
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Detects high-conviction mean-reversion opportunities when the price breaches a Bollinger Band,
    confirmed by extreme Money Flow Index (MFI) readings, higher-than-average volume, and a
    strong candle rejection pattern.
    """
    signals: list[BuySignal | SellSignal] = []
    rule_id = "35d6bda2-1656-4db1-872c-5a9b637e057a"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        if len(warm_candles) < MIN_CANDLES:
            continue

        # Extract data for calculations, ensuring they are numpy arrays
        # We need data up to the current candle for calculations
        closes = np.array([c.close for c in warm_candles])
        highs = np.array([c.high for c in warm_candles])
        lows = np.array([c.low for c in warm_candles])
        volumes = np.array([c.volume for c in warm_candles])
        
        # Ensure we have enough data for each specific calculation
        if len(closes) < PERIOD_BB or len(closes) < PERIOD_MFI + 1 or len(volumes) < PERIOD_VOL_AVG:
            continue

        # 1. Calculate Bollinger Bands (based on the last PERIOD_BB closes)
        sma_bb = _calculate_sma(closes, PERIOD_BB)
        std_dev_bb = _calculate_stddev(closes, PERIOD_BB)
        
        if np.isnan(sma_bb) or np.isnan(std_dev_bb):
            continue # Not enough data for BB

        upper_band = sma_bb + (std_dev_bb * STD_DEV_BB)
        lower_band = sma_bb - (std_dev_bb * STD_DEV_BB)

        # 2. Calculate Money Flow Index (MFI)
        mfi = _calculate_mfi(highs, lows, closes, volumes, PERIOD_MFI)
        if np.isnan(mfi):
            continue # Not enough data for MFI

        # 3. Calculate Average Volume
        avg_volume = _calculate_sma(volumes, PERIOD_VOL_AVG)
        if np.isnan(avg_volume):
            continue # Not enough data for Average Volume

        # Get the latest candle data for signal generation
        current_candle = warm_candles[-1]
        current_close = current_candle.close
        current_high = current_candle.high
        current_low = current_candle.low
        current_volume = current_candle.volume
        
        # Ensure high - low is not zero to avoid division by zero in rejection ratio
        candle_range = current_high - current_low
        if candle_range == 0:
            continue # Cannot determine rejection for a doji or zero-range candle

        # Check for Buy Signal
        buy_condition_bb = current_close < lower_band
        buy_condition_mfi = mfi < OVERSOLD_MFI
        buy_condition_volume = current_volume > (avg_volume * VOL_MULTIPLIER)
        
        # Bullish rejection candle: close significantly above low
        # (close - low) / (high - low) > rejection_ratio
        buy_condition_rejection = (current_close - current_low) / candle_range > REJECTION_RATIO

        if (buy_condition_bb and
            buy_condition_mfi and
            buy_condition_volume and
            buy_condition_rejection):
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_close,
                rule_id=rule_id
            ))

        # Check for Sell Signal
        sell_condition_bb = current_close > upper_band
        sell_condition_mfi = mfi > OVERBOUGHT_MFI
        sell_condition_volume = current_volume > (avg_volume * VOL_MULTIPLIER)
        
        # Bearish rejection candle: close significantly below high
        # (high - close) / (high - low) > rejection_ratio
        sell_condition_rejection = (current_high - current_close) / candle_range > REJECTION_RATIO

        if (sell_condition_bb and
            sell_condition_mfi and
            sell_condition_volume and
            sell_condition_rejection):
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour,
                price=current_close,
                rule_id=rule_id
            ))

    return signals