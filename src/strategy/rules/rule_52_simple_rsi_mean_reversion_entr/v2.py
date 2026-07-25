from datetime import datetime
import numpy as np
from src.agent.models import BuySignal, SellSignal, MarketData, WarmCandle, ColdMonth

# --- Constants ---
RSI_PERIOD = 14
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70
RSI_MID_RANGE_LOWER = 40
RSI_MID_RANGE_UPPER = 45

KCD_SHORT_THRESHOLD = 4
KCD_LONG_THRESHOLD = -1
PMMD_SHORT_THRESHOLD = 4
PMMD_LONG_THRESHOLD = -1

# Keltner Channel parameters (adjusted for 24-candle warm data constraint)
KCD_SMA_PERIOD = 10
KCD_ATR_PERIOD = 10

# Minimum data requirements for indicators
MIN_CANDLES_FOR_RSI = RSI_PERIOD + 1
MIN_CANDLES_FOR_KCD = max(KCD_SMA_PERIOD, KCD_ATR_PERIOD + 1)
MIN_WARM_CANDLES = max(MIN_CANDLES_FOR_RSI, MIN_CANDLES_FOR_KCD)


# --- Helper Functions ---
def _calculate_rsi(prices: list[float], period: int) -> list[float]:
    """
    Calculates the Relative Strength Index (RSI) for a given list of prices.
    """
    if len(prices) < period + 1:
        return []

    np_prices = np.array(prices)
    deltas = np_prices[1:] - np_prices[:-1]

    gains = np.maximum(0, deltas)
    losses = np.maximum(0, -deltas)

    avg_gains = np.zeros_like(gains)
    avg_losses = np.zeros_like(losses)

    avg_gains[period - 1] = np.mean(gains[:period])
    avg_losses[period - 1] = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period

    rs = np.divide(avg_gains[period - 1:], avg_losses[period - 1:],
                   out=np.full_like(avg_gains[period - 1:], np.inf),
                   where=avg_losses[period - 1:] != 0)

    rsi = 100 - (100 / (1 + rs))

    return rsi.tolist()


def _calculate_kcd(warm_candles: list[WarmCandle], sma_period: int, atr_period: int) -> float | None:
    """
    Calculates Keltner Channel Deviation (KCD) for the latest candle.
    KCD = (Close Price - SMA) / ATR
    """
    if len(warm_candles) < max(sma_period, atr_period + 1):
        return None

    closes = np.array([c.close for c in warm_candles])

    # Calculate SMA (centerline) for the latest `sma_period` candles
    sma = np.mean(closes[-sma_period:])

    # Calculate True Ranges
    true_ranges = []
    for i in range(1, len(warm_candles)):
        tr = max(
            warm_candles[i].high - warm_candles[i].low,
            abs(warm_candles[i].high - warm_candles[i-1].close),
            abs(warm_candles[i].low - warm_candles[i-1].close)
        )
        true_ranges.append(tr)

    # Calculate ATR (Simple Moving Average of True Ranges)
    if len(true_ranges) < atr_period:
        return None
    
    atr = np.mean(true_ranges[-atr_period:])

    if atr == 0:
        return 0.0

    latest_close = warm_candles[-1].close
    kcd = (latest_close - sma) / atr
    return kcd


def _calculate_pmmd(current_price: float, current_month_datetime: datetime, cold_months: list[ColdMonth]) -> float | None:
    """
    Calculates Price to Monthly Mean Deviation (PMMD).
    PMMD = (Current Close Price - Monthly Avg Price) / Monthly Avg Price
    """
    current_month_str = current_month_datetime.strftime("%Y-%m")
    monthly_avg_price = None

    for month_data in cold_months:
        if month_data.month == current_month_str:
            monthly_avg_price = month_data.avg_price
            break

    if monthly_avg_price is None or monthly_avg_price == 0:
        return None

    pmmd = (current_price - monthly_avg_price) / monthly_avg_price
    return pmmd


# --- Main Signal Function ---
def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    rule_id = "RSI_KCD_PMMD_Rev_Adj_1"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        cold_months = pair_data.cold

        if len(warm_candles) < MIN_WARM_CANDLES:
            continue

        latest_candle = warm_candles[-1]
        timestamp = latest_candle.hour
        price = latest_candle.close

        # Calculate RSI
        close_prices = [candle.close for candle in warm_candles]
        rsi_values = _calculate_rsi(close_prices, RSI_PERIOD)
        if not rsi_values:
            continue
        current_rsi = rsi_values[-1]

        # Calculate KCD
        current_kcd = _calculate_kcd(warm_candles, KCD_SMA_PERIOD, KCD_ATR_PERIOD)
        if current_kcd is None:
            continue

        # Calculate PMMD
        current_pmmd = _calculate_pmmd(price, timestamp, cold_months)
        if current_pmmd is None:
            continue

        # Short Entry Condition
        if (current_rsi > RSI_OVERBOUGHT_THRESHOLD and
            current_kcd > KCD_SHORT_THRESHOLD and
            current_pmmd > PMMD_SHORT_THRESHOLD):
            signals.append(SellSignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=rule_id
            ))

        # Long Entry Condition
        elif (current_rsi < RSI_OVERSOLD_THRESHOLD) or \
             (current_rsi >= RSI_MID_RANGE_LOWER and
              current_rsi <= RSI_MID_RANGE_UPPER and
              current_kcd < KCD_LONG_THRESHOLD and
              current_pmmd < PMMD_LONG_THRESHOLD):
            signals.append(BuySignal(
                pair=pair,
                timestamp=timestamp,
                price=price,
                rule_id=rule_id
            ))

    return signals