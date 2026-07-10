from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

# --- Rule Parameters ---
N_PRICE_RANGE = 10  # Lookback for resistance/support (hourly candles)
N_ATR = 14          # Lookback for current ATR (hourly candles)
N_ATR_AVG = 20      # Lookback for average ATR (hourly candles)

# N_VOLUME_AVG is for averaging volume from ticks, due to WarmCandle not having a 'volume' attribute.
# This averages the `volume_24h` from the last N_VOLUME_AVG ticks, which is a proxy for recent volume activity.
# This is a compromise and not ideal for candle-based volume confirmation as described in the pseudocode,
# which implies hourly volume. A more accurate implementation would require hourly volume in WarmCandle.
N_VOLUME_AVG = 60   # Lookback for average volume (ticks, not candles)

BREAKOUT_FACTOR = 0.01          # 1% breakout threshold
ATR_EXPANSION_FACTOR = 1.2      # 20% ATR expansion (e.g., current ATR is 20% higher than average)
VOLUME_CONFIRMATION_FACTOR = 1.5 # 50% volume increase (e.g., current volume is 50% higher than average)

# Minimum data requirements
MIN_CANDLES_FOR_CALC = max(N_PRICE_RANGE, N_ATR, N_ATR_AVG)
MIN_TICKS_FOR_VOLUME_AVG = N_VOLUME_AVG
MIN_TICKS_FOR_CURRENT_VOLUME = 1 # Need at least one tick for current volume_24h

def calculate_sma_atr(candles: list[WarmCandle], period: int) -> float | None:
    """Calculates the Simple Moving Average (SMA) based Average True Range (ATR)
    for the latest candle in the list over the specified period.
    
    Args:
        candles: A list of WarmCandle objects, ordered chronologically.
        period: The number of candles to use for the ATR calculation.

    Returns:
        The ATR value for the latest candle, or None if insufficient data.
    """
    if len(candles) < period:
        return None

    true_ranges_all = []
    for i in range(len(candles)):
        current = candles[i]
        
        # For the very first candle in the dataset, TR is High - Low
        # Otherwise, TR is max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
        if i == 0:
            tr = current.high - current.low
        else:
            prev_close = candles[i-1].close
            tr = max(current.high - current.low, abs(current.high - prev_close), abs(current.low - prev_close))
        true_ranges_all.append(tr)
    
    # The ATR for the *latest* candle is the SMA of the last `period` true ranges.
    # This means we need at least `period` true ranges calculated.
    if len(true_ranges_all) < period: # Should not happen if len(candles) >= period
        return None
    
    return statistics.mean(true_ranges_all[-period:])


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the Volatility-Adjusted Breakout with Volume Confirmation rule.

    This rule identifies potential price breakouts from a consolidation phase,
    confirmed by an increase in volatility and higher-than-average trading volume.
    It generates a Buy signal when the price breaks above a recent resistance level,
    accompanied by a significant increase in both volatility (e.g., ATR expansion)
    and volume. Conversely, it generates a Sell signal when the price breaks below
    a recent support level under similar conditions.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Check for sufficient candle data for price range and ATR calculations
        if len(warm_candles) < MIN_CANDLES_FOR_CALC:
            continue

        # Check for sufficient tick data for volume calculations
        if len(hot_ticks) < MIN_TICKS_FOR_VOLUME_AVG or len(hot_ticks) < MIN_TICKS_FOR_CURRENT_VOLUME:
            continue

        current_candle = warm_candles[-1]
        current_close = current_candle.close

        # 1. Lookback periods defined as constants.

        # 2. Calculate recent High and Low over N_PRICE_RANGE to define resistance and support levels.
        # Resistance = MAX(High, N_price_range); Support = MIN(Low, N_price_range).
        price_range_candles = warm_candles[-N_PRICE_RANGE:]
        resistance = max(c.high for c in price_range_candles)
        support = min(c.low for c in price_range_candles)

        # 3. Calculate current Average True Range (ATR) over N_ATR.
        current_atr = calculate_sma_atr(warm_candles, N_ATR)
        if current_atr is None: # Should not happen if MIN_CANDLES_FOR_CALC is met
            continue

        # 4. Calculate the average ATR over a longer period (N_ATR_AVG).
        average_atr_long_period = calculate_sma_atr(warm_candles, N_ATR_AVG)
        if average_atr_long_period is None: # Should not happen if MIN_CANDLES_FOR_CALC is met
            continue
        
        # Determine ATR expansion condition
        atr_expansion_condition = False
        if average_atr_long_period > 0: # Avoid division by zero and ensure meaningful expansion
            atr_expansion_condition = (current_atr > average_atr_long_period * ATR_EXPANSION_FACTOR)
        # If average_atr_long_period is 0, it implies no volatility, so no expansion can be confirmed.

        # 5. Calculate current Volume and the average Volume over a longer period (N_VOLUME_AVG).
        # Note: Using volume_24h from ticks as a proxy due to WarmCandle model limitations.
        current_volume = hot_ticks[-1].volume_24h
        
        past_volumes_24h = [t.volume_24h for t in hot_ticks[-N_VOLUME_AVG:]]
        average_volume_long_period = statistics.mean(past_volumes_24h)
        
        # Determine volume confirmation condition
        volume_confirmation_condition = False
        if average_volume_long_period > 0: # Avoid division by zero and ensure meaningful confirmation
            volume_confirmation_condition = (current_volume > average_volume_long_period * VOLUME_CONFIRMATION_FACTOR)
        # If average_volume_long_period is 0, it implies no trading, so no confirmation can be confirmed.

        # 7. Buy Signal Conditions:
        # IF Current_Close > Resistance * (1 + breakout_factor)
        # AND Current_ATR > Average_ATR_over_N_atr_avg * atr_expansion_factor
        # AND Current_Volume > Average_Volume_over_N_volume_avg * volume_confirmation_factor
        if (current_close > resistance * (1 + BREAKOUT_FACTOR) and
            atr_expansion_condition and
            volume_confirmation_condition):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour, # Use candle close time for signal timestamp
                price=current_close,
                rule_id="6e6f0221-dea4-4731-aefd-a6dd9799950c",
                confidence=None # Confidence could be derived from strength of breakout/confirmation factors
            ))

        # 8. Sell Signal Conditions:
        # IF Current_Close < Support * (1 - breakout_factor)
        # AND Current_ATR > Average_ATR_over_N_atr_avg * atr_expansion_factor
        # AND Current_Volume > Average_Volume_over_N_volume_avg * volume_confirmation_factor
        elif (current_close < support * (1 - BREAKOUT_FACTOR) and
              atr_expansion_condition and
              volume_confirmation_condition):
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour, # Use candle close time for signal timestamp
                price=current_close,
                rule_id="6e6f0221-dea4-4731-aefd-a6dd9799950c",
                confidence=None # Confidence could be derived from strength of breakout/confirmation factors
            ))

    return signals