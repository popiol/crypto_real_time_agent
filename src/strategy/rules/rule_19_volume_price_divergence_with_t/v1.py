from __future__ import annotations
import statistics
import numpy as np
from datetime import datetime, timedelta
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick

# --- Rule Parameters ---
# Periods for price moving averages (based on hourly candles)
SMA_SHORT_PERIOD_PRICE = 5
SMA_MEDIUM_PERIOD_PRICE = 10

# Period for volume moving average (based on ticks)
SMA_SHORT_PERIOD_VOLUME = 5

# Minimum data requirements
# For price SMAs, we need at least SMA_MEDIUM_PERIOD_PRICE candles
# For divergence, we need at least 2 candles (current and previous)
MIN_CANDLES_REQUIRED = max(2, SMA_MEDIUM_PERIOD_PRICE)

# For volume SMAs, we need SMA_SHORT_PERIOD_VOLUME + 1 ticks to have at least two SMA values
# (current and previous) for divergence comparison.
MIN_TICKS_REQUIRED = SMA_SHORT_PERIOD_VOLUME + 1

# --- Helper function for Simple Moving Average (SMA) ---
def calculate_sma(data: list[float], period: int) -> list[float]:
    """
    Calculates the Simple Moving Average (SMA) for a given list of data.
    Returns a list of SMA values, where each value is the SMA ending at that point.
    Uses 'valid' mode for convolution, meaning the output will be shorter than input.
    """
    if len(data) < period:
        return []
    # Using numpy for efficient moving average calculation
    sma_values = np.convolve(data, np.ones(period) / period, mode='valid').tolist()
    return sma_values

# --- Main signal generation function ---
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Volume-Price Divergence with Trend Confirmation' trading rule.

    This rule detects bearish or bullish divergences between price action (using
    hourly candles) and trading volume (using tick-level 24h rolling volume),
    confirmed by a medium-term price trend indicator.

    A Buy signal is generated when price makes lower lows but volume shows higher
    lows (bullish divergence), occurring within an overall uptrend.
    A Sell signal is generated when price makes higher highs but volume shows
    lower highs (bearish divergence), occurring within an overall downtrend.

    Note on Volume Data: The WarmCandle model lacks a volume attribute. This
    implementation uses the 'volume_24h' from the most recent 'Tick' data in
    'pair_data.hot' to infer current and previous volume levels. This is a
    pragmatic compromise given the available data models and may not perfectly
    align volume with specific hourly candle price movements but serves as
    a proxy for overall recent volume activity.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # --- Data Validation ---
        # Ensure sufficient historical candles for price SMAs and divergence comparison
        if len(warm_candles) < MIN_CANDLES_REQUIRED:
            continue
        
        # Ensure sufficient recent ticks for volume SMAs
        if len(hot_ticks) < MIN_TICKS_REQUIRED:
            continue

        # --- 1. Prepare Price Data (from WarmCandle) ---
        # Extract close prices for SMA calculations
        close_prices = [c.close for c in warm_candles]
        
        # Get current and previous candle data for divergence detection
        current_candle = warm_candles[-1]
        previous_candle = warm_candles[-2]
        
        current_low = current_candle.low
        previous_low = previous_candle.low
        
        current_high = current_candle.high
        previous_high = previous_candle.high
        
        # --- 2. Prepare Volume Data (from Tick) ---
        # Extract the 24-hour rolling volume from recent ticks.
        # This serves as a proxy for market activity volume.
        tick_volumes = [t.volume_24h for t in hot_ticks]
        
        # --- 3. Calculate Simple Moving Averages (SMAs) ---
        # Price SMAs (calculated on hourly candle close prices)
        sma_short_price_series = calculate_sma(close_prices, SMA_SHORT_PERIOD_PRICE)
        sma_medium_price_series = calculate_sma(close_prices, SMA_MEDIUM_PERIOD_PRICE)

        # Volume SMA (calculated on tick-level 24h rolling volume)
        sma_short_volume_series = calculate_sma(tick_volumes, SMA_SHORT_PERIOD_VOLUME)
        
        # Ensure all required SMA series have been successfully calculated
        if not sma_short_price_series or not sma_medium_price_series or not sma_short_volume_series:
            continue

        # Get the latest SMA values for trend confirmation
        current_sma_short_price = sma_short_price_series[-1]
        current_sma_medium_price = sma_medium_price_series[-1]
        
        # Get the latest and previous SMA volume values for divergence comparison.
        # We need at least two values in the SMA series to compare current vs. previous.
        if len(sma_short_volume_series) < 2:
            continue

        current_avg_volume = sma_short_volume_series[-1]
        previous_avg_volume = sma_short_volume_series[-2]

        # --- 4. Identify Bullish Divergence ---
        # Conditions:
        #   1. Price makes a lower low (comparing current and previous hourly candle lows).
        #   2. Volume shows a higher low (comparing current and previous tick-based volume SMAs).
        #   3. Confirmed by an overall uptrend (short-term price SMA > medium-term price SMA).
        if (current_low < previous_low and                      # Lower low in price
            current_avg_volume > previous_avg_volume and        # Higher low in volume (SMA-based)
            current_sma_short_price > current_sma_medium_price): # Overall uptrend (price SMA)
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_candle.hour, # Signal timestamp is the hour of the latest candle
                price=current_candle.close,
                rule_id="feb622ec-ad8c-4744-8d83-998875053ca4",
                confidence=0.7 # Placeholder confidence value
            ))

        # --- 5. Identify Bearish Divergence ---
        # Conditions:
        #   1. Price makes a higher high (comparing current and previous hourly candle highs).
        #   2. Volume shows a lower high (comparing current and previous tick-based volume SMAs).
        #   3. Confirmed by an overall downtrend (short-term price SMA < medium-term price SMA).
        elif (current_high > previous_high and                   # Higher high in price
              current_avg_volume < previous_avg_volume and       # Lower high in volume (SMA-based)
              current_sma_short_price < current_sma_medium_price): # Overall downtrend (price SMA)
            
            signals.append(SellSignal(
                pair=pair,
                timestamp=current_candle.hour, # Signal timestamp is the hour of the latest candle
                price=current_candle.close,
                rule_id="feb622ec-ad8c-4744-8d83-998875053ca4",
                confidence=0.7 # Placeholder confidence value
            ))

    return signals