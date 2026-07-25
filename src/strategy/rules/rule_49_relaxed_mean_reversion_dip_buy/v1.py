from __future__ import annotations
import statistics
from datetime import datetime
from src.agent.models import BuySignal, MarketData, SellSignal, Tick, WarmCandle

# --- Parameters ---
SHORT_MA_PERIOD = 20
MEDIUM_MA_PERIOD = 50
ATR_PERIOD = 14
ATR_MA_PERIOD = 10
MAX_SPREAD_PERCENT = 0.002  # 0.2% as a decimal
MIN_LIQUIDITY_DEPTH = 100000.0 # Notional value, using float for consistency

# Minimum candles required for all calculations:
# - For SMA_SHORT: SHORT_MA_PERIOD
# - For SMA_MEDIUM: MEDIUM_MA_PERIOD
# - For ATR series: ATR_PERIOD candles (to get the first ATR value which is an SMA of TRs)
# - For ATR_SMA: ATR_MA_PERIOD ATR values.
#   Since the ATR series itself is derived from candles, we need enough candles
#   to generate `ATR_MA_PERIOD` values in the `atr_series`.
#   The `_calculate_atr_series` function produces `len(candles) - ATR_PERIOD + 1` ATR values.
#   So, we need `len(candles) - ATR_PERIOD + 1 >= ATR_MA_PERIOD`, which simplifies to
#   `len(candles) >= ATR_PERIOD + ATR_MA_PERIOD - 1`.
#   Example: 14 + 10 - 1 = 23 candles.
#   Taking the maximum of all requirements:
MIN_CANDLES_REQUIRED = max(SHORT_MA_PERIOD, MEDIUM_MA_PERIOD, ATR_PERIOD + ATR_MA_PERIOD - 1)
# For the given parameters: max(20, 50, 14 + 10 - 1) = max(20, 50, 23) = 50.
MIN_TICKS_REQUIRED = 1 # Need at least one tick for current market data


# --- Helper Functions for Indicators ---

def _calculate_sma(data: list[float], period: int) -> float | None:
    """
    Calculates the Simple Moving Average for the last 'period' values in the given data list.
    Returns None if there is insufficient data.
    """
    if len(data) < period:
        return None
    return statistics.mean(data[-period:])

def _calculate_true_ranges(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    """
    Calculates True Range (TR) for each candle in a series.
    TR[i] = max(high[i] - low[i], abs(high[i] - closes[i-1]), abs(low[i] - closes[i-1]))
    For the first candle (index 0), TR is high[0] - low[0] as no previous close exists.
    """
    true_ranges = []
    if not closes:
        return true_ranges
    
    # TR for the very first candle
    true_ranges.append(highs[0] - lows[0])

    # TR for subsequent candles
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        true_ranges.append(tr)
    return true_ranges

def _calculate_atr_series(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    """
    Calculates the Average True Range (ATR) series.
    Each ATR value in the series is a Simple Moving Average of the True Ranges
    over the specified period. This interpretation allows for an SMA of ATR values.
    """
    tr_values = _calculate_true_ranges(highs, lows, closes)
    
    if len(tr_values) < period:
        return []

    atr_series = []
    # Calculate SMA of TRs for each point from 'period' onwards
    for i in range(period - 1, len(tr_values)):
        atr_series.append(statistics.mean(tr_values[i - period + 1 : i + 1]))
    
    return atr_series


# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Relaxed Mean-Reversion Dip Buy Strategy' trading rule.
    Initiates a long position when the asset's close price dips below both a short-term
    (20-period) and a medium-term (50-period) Simple Moving Average. This entry is
    conditional on moderately elevated volatility (ATR_14 > SMA_10(ATR_14)) and
    acceptable market liquidity (spread < 0.2% and notional depth > $100,000).
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # 1. Data Sufficiency Check
        if len(warm_candles) < MIN_CANDLES_REQUIRED or len(hot_ticks) < MIN_TICKS_REQUIRED:
            continue

        # Extract candle data for calculations
        closes = [c.close for c in warm_candles]
        highs = [c.high for c in warm_candles]
        lows = [c.low for c in warm_candles]

        # 2. Calculate Indicators

        # Simple Moving Averages for price
        sma_short = _calculate_sma(closes, SHORT_MA_PERIOD)
        sma_medium = _calculate_sma(closes, MEDIUM_MA_PERIOD)

        if sma_short is None or sma_medium is None:
            # This check is technically redundant if MIN_CANDLES_REQUIRED is correct,
            # but serves as a robust fallback.
            continue

        # ATR series (SMA of True Ranges)
        atr_series = _calculate_atr_series(highs, lows, closes, ATR_PERIOD)
        if not atr_series:
            continue # Not enough data to calculate ATR series

        current_atr = atr_series[-1] # The latest ATR value

        # SMA of the ATR series
        atr_sma = _calculate_sma(atr_series, ATR_MA_PERIOD)
        if atr_sma is None:
            # Not enough ATR values to calculate SMA of ATR
            continue

        # Current tick data for real-time conditions
        current_tick = hot_ticks[-1]
        current_close_price = current_tick.last_price
        
        # Ensure mid_price is not zero to avoid division by zero
        if current_tick.mid_price == 0:
            continue
        
        # Convert spread_rel (percentage) to decimal
        spread_percent = current_tick.spread_rel / 100.0
        
        # Calculate notional liquidity depth
        liquidity_depth = (current_tick.bid_volume * current_tick.bid_price) + \
                          (current_tick.ask_volume * current_tick.ask_price)

        # 3. Check Entry Condition (Long)
        price_dip_condition = (current_close_price < sma_short) and \
                              (current_close_price < sma_medium)
        
        volatility_condition = (current_atr > atr_sma)
        
        liquidity_condition = (spread_percent < MAX_SPREAD_PERCENT) and \
                              (liquidity_depth > MIN_LIQUIDITY_DEPTH)

        if price_dip_condition and volatility_condition and liquidity_condition:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=current_close_price,
                rule_id="relaxed_dip_buy_v1",
                confidence=0.7 # A default confidence for a proposed rule
            ))

    return signals