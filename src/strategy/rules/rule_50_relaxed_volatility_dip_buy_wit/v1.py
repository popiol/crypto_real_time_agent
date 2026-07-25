from __future__ import annotations
import statistics # Not strictly used in this rule, but common for such modules.
from datetime import datetime
from src.agent.models import BuySignal, MarketData, WarmCandle, Tick, SellSignal

# Rule Constants
SMA_PERIOD = 50
ATR_PERIOD = 14
ATR_MA_PERIOD = 20
DIP_THRESHOLD_PERCENT = 0.02 # 2%
MAX_SPREAD_PERCENT = 0.001 # 0.1%
ATR_MULTIPLIER = 1.5

# Minimum data required for calculations
MIN_TICKS_FOR_LIQUIDITY = 1
MIN_CANDLES_FOR_SMA = SMA_PERIOD
# To calculate SMA_OF_ATR, we need ATR_MA_PERIOD ATR values.
# Each ATR value needs ATR_PERIOD true ranges.
# To generate 'K' true ranges, we need 'K+1' candles.
# To get ATR_MA_PERIOD ATR values, the `_calculate_atr_series` function
# needs (ATR_MA_PERIOD - 1) + ATR_PERIOD true ranges to be available for the last ATR_MA_PERIOD ATRs.
# This means we need at least (ATR_MA_PERIOD - 1) + ATR_PERIOD + 1 candles.
# So, MIN_CANDLES_FOR_ATR_CALCS = ATR_PERIOD + ATR_MA_PERIOD
MIN_CANDLES_FOR_ATR_CALCS = ATR_PERIOD + ATR_MA_PERIOD

# Overall minimum candles required for all calculations
MIN_CANDLES_REQUIRED = max(MIN_CANDLES_FOR_SMA, MIN_CANDLES_FOR_ATR_CALCS)


def _calculate_sma(data: list[float], period: int) -> float | None:
    """
    Calculates the Simple Moving Average for the last 'period' values in 'data'.
    Returns None if insufficient data.
    """
    if len(data) < period:
        return None
    return sum(data[-period:]) / period

def _calculate_true_range(high: float, low: float, close_prev: float) -> float:
    """
    Calculates the True Range for a given candle using its high, low,
    and the previous candle's close price.
    """
    range1 = high - low
    range2 = abs(high - close_prev)
    range3 = abs(low - close_prev)
    return max(range1, range2, range3)

def _calculate_atr_series(candles: list[WarmCandle], period: int) -> list[float]:
    """
    Calculates a series of ATR values (Simple Moving Average of True Range).
    Each ATR_t is the SMA of 'period' True Ranges ending at candle_t.
    Requires at least `period + 1` candles to produce the first ATR value.
    Returns an empty list if insufficient data.
    """
    if len(candles) < 2: # Need at least two candles to calculate the first TR
        return []

    true_ranges = []
    # Calculate True Range for candles[1] to candles[-1]
    for i in range(1, len(candles)):
        tr = _calculate_true_range(
            high=candles[i].high,
            low=candles[i].low,
            close_prev=candles[i-1].close
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return [] # Not enough true ranges to calculate even one ATR

    atr_values = []
    # Calculate SMA of true_ranges for each point where `period` true ranges are available
    for i in range(period - 1, len(true_ranges)):
        current_atr = sum(true_ranges[i - period + 1 : i + 1]) / period
        atr_values.append(current_atr)
    return atr_values


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates a buy signal when the asset's closing price dips at least 2% below its
    50-period Simple Moving Average (SMA), while current volatility (14-period ATR)
    is at least 1.5 times its 20-period SMA of ATR, and the bid-ask spread is less
    than 0.1% of the mid-price.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        candles = pair_data.warm

        # --- Data Sufficiency Checks ---
        if len(ticks) < MIN_TICKS_FOR_LIQUIDITY:
            continue
        if len(candles) < MIN_CANDLES_REQUIRED:
            continue

        # --- Extract Current Data ---
        current_tick = ticks[-1]
        current_candle = candles[-1]
        current_close = current_candle.close

        bid_price = current_tick.bid_price
        ask_price = current_tick.ask_price
        mid_price = current_tick.mid_price

        # Ensure mid_price is positive to avoid division by zero or negative prices
        if mid_price <= 0:
            continue

        # --- Calculate Indicators ---
        # 1. SMA of Close Price
        close_prices = [c.close for c in candles]
        sma_price = _calculate_sma(close_prices, SMA_PERIOD)
        if sma_price is None or sma_price <= 0: # SMA must be positive for meaningful comparison
            continue

        # 2. ATR Series and SMA of ATR
        atr_series = _calculate_atr_series(candles, ATR_PERIOD)
        # We need enough ATR values to calculate the SMA_OF_ATR
        if len(atr_series) < ATR_MA_PERIOD:
            continue

        current_atr = atr_series[-1]
        sma_of_atr = _calculate_sma(atr_series, ATR_MA_PERIOD)

        # Ensure ATR values are positive for meaningful ratios
        if sma_of_atr is None or sma_of_atr <= 0:
            continue
        if current_atr <= 0:
            continue

        # --- Apply Conditions ---

        # CONDITION 1: Price Dip
        # Current close price dips at least DIP_THRESHOLD_PERCENT below its SMA_PERIOD SMA
        condition_1_price_dip = current_close < (sma_price * (1 - DIP_THRESHOLD_PERCENT))

        # CONDITION 2: Volatility
        # Current volatility (ATR_PERIOD ATR) is at least ATR_MULTIPLIER times its ATR_MA_PERIOD SMA of ATR
        condition_2_volatility = current_atr > (sma_of_atr * ATR_MULTIPLIER)

        # CONDITION 3: Liquidity
        # Bid-ask spread is less than MAX_SPREAD_PERCENT of the mid-price
        condition_3_liquidity = (ask_price - bid_price) / mid_price < MAX_SPREAD_PERCENT

        # --- Generate Signal ---
        if condition_1_price_dip and condition_2_volatility and condition_3_liquidity:
            signals.append(BuySignal(
                pair=pair,
                timestamp=current_tick.polled_at,
                price=current_tick.last_price,
                rule_id="RelaxedDipBuy_v1"
            ))

    return signals