from __future__ import annotations
import statistics
import numpy as np
from src.agent.models import BuySignal, SellSignal, PairData, Tick

RULE_ID = "rule_01_spread_compression_v2"

# Parameters from pseudocode
SPREAD_EMA_PERIOD = 20
SPREAD_STD_PERIOD = 20
STD_MULTIPLIER = 1.5
VOLUME_INCREASE_FACTOR = 1.5
VOLUME_EMA_PERIOD = 50

# Minimum ticks required for all calculations
# This ensures that all EMA/STDDEV periods can be fully calculated for the latest tick.
MIN_TICKS_REQUIRED = max(SPREAD_EMA_PERIOD, SPREAD_STD_PERIOD, VOLUME_EMA_PERIOD)

MarketData = dict[str, PairData]

def _calculate_ema(data_series: list[float], period: int) -> float:
    """
    Calculates the Exponential Moving Average for the last element in the series.
    Assumes len(data_series) >= period.
    """
    # Calculate SMA for the initial period to seed the EMA
    sma = np.mean(data_series[:period])
    
    ema = sma
    alpha = 2 / (period + 1)
    
    # Iterate from the first element AFTER the initial SMA period to calculate subsequent EMAs
    for i in range(period, len(data_series)):
        ema = (data_series[i] * alpha) + (ema * (1 - alpha))
        
    return ema

def _calculate_stddev(data_series: list[float], period: int) -> float:
    """
    Calculates the Standard Deviation of the last 'period' elements in the series.
    Assumes len(data_series) >= period.
    """
    subset = data_series[-period:]
    # np.std with default ddof=0 calculates population standard deviation.
    return np.std(subset)

def spread_compression_v2(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        
        # Ensure we have enough historical data for all required calculations
        if len(ticks) < MIN_TICKS_REQUIRED:
            continue

        # Extract required data series from the Tick objects
        # t.spread_rel is (ask_price - bid_price) / last_price
        spreads = [t.spread_rel for t in ticks]
        volumes = [t.volume for t in ticks] 

        # The last tick represents the current market state
        current_spread = spreads[-1]
        current_volume = volumes[-1]
        
        # Calculate EMAs and STDDEV for the current state using historical data
        # These functions are guaranteed to have enough data due to MIN_TICKS_REQUIRED check,
        # so they will not return None.
        spread_ema = _calculate_ema(spreads, SPREAD_EMA_PERIOD)
        spread_std = _calculate_stddev(spreads, SPREAD_STD_PERIOD)
        volume_ema = _calculate_ema(volumes, VOLUME_EMA_PERIOD)

        # Define adaptive bands based on the spread's EMA and historical standard deviation
        upper_band = spread_ema + (spread_std * STD_MULTIPLIER)
        lower_band = spread_ema - (spread_std * STD_MULTIPLIER)

        # Check for Buy Signal: spread compression beyond the adaptive lower band 
        # with confirmation from a significant increase in trading volume.
        if current_spread < lower_band and current_volume > (volume_ema * VOLUME_INCREASE_FACTOR):
            signals.append(BuySignal(
                pair=pair,
                rule_id=RULE_ID,
                timestamp=ticks[-1].polled_at,
                price=ticks[-1].last_price,
            ))

        # Check for Sell Signal: spread expansion beyond the adaptive upper band 
        # with confirmation from a significant increase in trading volume.
        if current_spread > upper_band and current_volume > (volume_ema * VOLUME_INCREASE_FACTOR):
            signals.append(SellSignal(
                pair=pair,
                rule_id=RULE_ID,
                timestamp=ticks[-1].polled_at,
                price=ticks[-1].last_price,
            ))

    return signals