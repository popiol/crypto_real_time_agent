"""Rule 15afb274-c4f0-45d0-b1d6-ba124e77b3d7 — Bollinger Band Breakout with Volume Confirmation."""
from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, PairData, SellSignal

# Constants for Bollinger Bands calculation
N_BB_PERIODS = 10  # Number of periods for Bollinger Band SMA and STD (using warm candles)
K = 2.0            # Multiplier for Standard Deviation to define bands

# Constants for Volume Confirmation
M_VOLUME_SMA_PERIODS = 30 # Number of periods for Volume SMA (using hot ticks)
VOLUME_MULTIPLIER = 1.5   # Multiplier for average volume to confirm breakout (e.g., 1.5 means 50% higher than average)

# Minimum data requirements
# We need at least N_BB_PERIODS warm candles for Bollinger Bands
# And at least M_VOLUME_SMA_PERIODS hot ticks for volume SMA and current tick data
MIN_CANDLES_FOR_BB = N_BB_PERIODS
MIN_TICKS_FOR_VOLUME_SMA = M_VOLUME_SMA_PERIODS


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure sufficient warm candle data for Bollinger Bands calculation
        if len(pair_data.warm) < MIN_CANDLES_FOR_BB:
            continue

        # Ensure sufficient hot tick data for current price, timestamp, and volume SMA calculation
        if len(pair_data.hot) < MIN_TICKS_FOR_VOLUME_SMA:
            continue

        # 1. Calculate Bollinger Bands using closing prices from warm candles
        # Use the most recent N_BB_PERIODS warm candles
        bb_closes = [c.close for c in pair_data.warm[-N_BB_PERIODS:]]
        
        # Calculate Simple Moving Average (SMA) of closing prices
        bb_mean = statistics.mean(bb_closes)
        # Calculate Standard Deviation (STD) of closing prices
        bb_std = statistics.stdev(bb_closes)

        # If standard deviation is zero, prices have not moved, bands are flat.
        # No meaningful breakout can occur, so skip this pair.
        if bb_std == 0:
            continue

        # Calculate Upper and Lower Bollinger Bands
        upper_bollinger_band = bb_mean + (K * bb_std)
        lower_bollinger_band = bb_mean - (K * bb_std)

        # 2. Calculate Simple Moving Average of Volume using 'volume_24h' from hot ticks
        # Use the most recent M_VOLUME_SMA_PERIODS hot ticks for volume average
        volume_24h_values = [t.volume_24h for t in pair_data.hot[-M_VOLUME_SMA_PERIODS:]]
        
        # Calculate SMA of the 24-hour rolling volume
        sma_volume = statistics.mean(volume_24h_values)

        # If average volume is zero, no meaningful volume confirmation can occur.
        # This might indicate an inactive pair or data issues.
        if sma_volume == 0:
            continue
            
        # 3. Get current market data from the most recent tick
        current_tick = pair_data.hot[-1]
        current_price = current_tick.last_price
        current_volume_24h = current_tick.volume_24h
        timestamp = current_tick.polled_at

        # 4. Emit Signals based on Bollinger Band breakout with Volume Confirmation
        # Buy signal: Current price drops below the Lower Bollinger Band
        # AND current 24-hour volume is significantly higher than its recent average.
        if current_price < lower_bollinger_band and \
           current_volume_24h > (VOLUME_MULTIPLIER * sma_volume):
            signals.append(BuySignal(pair=pair, timestamp=timestamp, price=current_price))
        
        # Sell signal: Current price rises above the Upper Bollinger Band
        # AND current 24-hour volume is significantly higher than its recent average.
        elif current_price > upper_bollinger_band and \
             current_volume_24h > (VOLUME_MULTIPLIER * sma_volume):
            signals.append(SellSignal(pair=pair, timestamp=timestamp, price=current_price))

    return signals