from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal

# Rule constants
RULE_ID = "HVDipBuy_V2"

# Thresholds from the rule description
MRO_BUY_THRESHOLD = -500.0
SPREAD_TO_DEPTH_BUY_THRESHOLD = 0.0001
BB_WIDTH_BUY_THRESHOLD = 0.03

# Lookback periods for calculations
N_WARM_PERIOD = 20 # For SMA, StdDev, BB Width, MRO, using hourly candles
MIN_HOT_TICKS = 1 # Need at least one tick for current spread/depth

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        hot_ticks = pair_data.hot
        warm_candles = pair_data.warm

        # --- Data sufficiency checks ---
        if len(hot_ticks) < MIN_HOT_TICKS:
            continue
        # Need enough warm candles for SMA/StdDev calculations
        if len(warm_candles) < N_WARM_PERIOD:
            continue

        # --- Calculate Hot_Spread_to_Depth_Ratio ---
        last_tick = hot_ticks[-1]
        
        current_depth = last_tick.bid_volume + last_tick.ask_volume
        if current_depth == 0:
            # If no depth, liquidity is effectively zero. A high ratio will fail the condition.
            hot_spread_to_depth_ratio = float('inf')
        else:
            # last_tick.spread_abs is available from the Tick model description.
            hot_spread_to_depth_ratio = last_tick.spread_abs / current_depth

        # --- Calculate Warm_Mean_Reversion_Oscillator and Warm_Bollinger_Band_Width ---
        # Use the most recent N_WARM_PERIOD candles for calculations
        relevant_warm_candles = warm_candles[-N_WARM_PERIOD:]
        close_prices = [c.close for c in relevant_warm_candles]

        # statistics.stdev requires at least 2 data points.
        if len(close_prices) < 2:
            continue

        try:
            sma_warm = statistics.mean(close_prices)
            stddev_warm = statistics.stdev(close_prices)
        except statistics.StatisticsError:
            # This can happen if all `close_prices` are identical, leading to 0 standard deviation.
            # Or if len(close_prices) was 1, which is caught by the previous check.
            # If all prices are identical, stddev is 0, which is handled correctly below.
            # In other cases of error (e.g., malformed data), we default stddev to 0.0.
            stddev_warm = 0.0

        # Warm Mean Reversion Oscillator: (last_close - SMA)
        # This measures how far the current price is below the moving average.
        warm_mean_reversion_oscillator = close_prices[-1] - sma_warm

        # Warm Bollinger Band Width: (4 * StdDev) / SMA (assuming K=2 for BBands)
        # This is a relative measure of volatility.
        if sma_warm == 0:
            # If SMA is zero (all prices are zero), this ratio is undefined.
            # Setting to 0.0 means no volatility, which will fail the 'greater than' threshold.
            warm_bollinger_band_width = 0.0
        else:
            warm_bollinger_band_width = (4 * stddev_warm) / sma_warm

        # --- Apply the trading rule conditions ---
        if (warm_mean_reversion_oscillator < MRO_BUY_THRESHOLD and
                hot_spread_to_depth_ratio < SPREAD_TO_DEPTH_BUY_THRESHOLD and
                warm_bollinger_band_width > BB_WIDTH_BUY_THRESHOLD):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=last_tick.polled_at,
                price=last_tick.last_price,
                rule_id=RULE_ID
            ))

    return signals