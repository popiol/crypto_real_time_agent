from __future__ import annotations
import numpy as np
from datetime import datetime
from src.agent.models import BuySignal, MarketData, WarmCandle, Tick, SellSignal

# Indicator calculation helper functions

def calculate_rsi(warm_candles: list[WarmCandle], period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for the last candle.
    Uses a simple moving average for gains and losses over the period.
    """
    closes = np.array([c.close for c in warm_candles])

    # We need 'period' price changes, which requires 'period + 1' close prices.
    if len(closes) < period + 1:
        return np.nan

    # Get the last 'period + 1' closes to calculate the last 'period' changes.
    recent_closes = closes[-(period + 1):]
    changes = np.diff(recent_closes)

    gains = changes[changes > 0]
    losses = -changes[changes < 0]

    avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    if avg_loss == 0:
        # If there are no losses, RSI is 100 (if there were gains) or 50 (if flat).
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands_width_percentage(warm_candles: list[WarmCandle], period: int) -> float:
    """
    Calculates Bollinger Bands Width Percentage (BBW) for the last candle.
    Uses Simple Moving Average (SMA) and Standard Deviation (STDDEV).
    """
    closes = np.array([c.close for c in warm_candles])

    # We need at least 'period' closes for SMA and STDDEV.
    if len(closes) < period:
        return np.nan

    # Use the last 'period' closes for calculation.
    recent_closes = closes[-period:]

    sma = np.mean(recent_closes)
    std = np.std(recent_closes)

    if sma == 0: # Avoid division by zero if prices are extremely low or zero.
        return np.nan

    upper_band = sma + (std * 2) # Standard deviation multiplier is typically 2
    lower_band = sma - (std * 2)

    bbw = ((upper_band - lower_band) / sma) * 100
    return bbw

def calculate_keltner_channel_percentage(warm_candles: list[WarmCandle], period: int) -> float:
    """
    Calculates Keltner Channel Percentage (KCP) for the last candle.
    Uses ATR for channel width and SMA of closes for the middle line.
    """
    highs = np.array([c.high for c in warm_candles])
    lows = np.array([c.low for c in warm_candles])
    closes = np.array([c.close for c in warm_candles])

    # For ATR, we need 'period' True Range values. Each TR value requires the
    # current candle's H, L and the previous candle's C. So, to get 'period'
    # TRs for the last 'period' candles, we need 'period + 1' candles in total.
    if len(warm_candles) < period + 1:
        return np.nan

    # Calculate True Range (TR) for all available candles starting from the second one.
    # TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
    all_true_ranges = []
    for i in range(1, len(warm_candles)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        all_true_ranges.append(tr)

    # We need at least 'period' True Range values to calculate ATR.
    if len(all_true_ranges) < period:
        return np.nan

    # ATR is the Simple Moving Average of the last 'period' True Range values.
    atr = np.mean(all_true_ranges[-period:])

    # Middle Line (ML) uses SMA of the last 'period' close prices.
    ml = np.mean(closes[-period:])

    multiplier = 2 # Common multiplier for Keltner Channels

    upper_channel = ml + (multiplier * atr)
    lower_channel = ml - (multiplier * atr)

    # Avoid division by zero if the channel width is zero (implies no volatility).
    if (upper_channel - lower_channel) == 0:
        return np.nan

    # KCP for the last close price.
    kcp = ((closes[-1] - lower_channel) / (upper_channel - lower_channel)) * 100
    return kcp


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    rule_id = "uptrend_liquidity_filter"

    # Define periods for indicators as per the rule description.
    rsi_period = 14
    bb_period = 20
    kc_period = 20

    # Determine the minimum number of warm candles required for all indicators.
    # RSI(14) needs 14 changes -> 15 candles.
    # BBW(20) needs 20 closes -> 20 candles.
    # KCP(20) needs 20 TR values (each using prev_close) -> 21 candles.
    min_warm_candles = max(rsi_period + 1, bb_period, kc_period + 1) # This evaluates to 21.

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Initial check for sufficient warm data.
        if len(warm_candles) < min_warm_candles:
            continue

        # Check for sufficient hot data to calculate spread.
        if not hot_ticks:
            continue
        latest_tick = hot_ticks[-1]

        # Calculate indicators.
        rsi = calculate_rsi(warm_candles, rsi_period)
        bbw = calculate_bollinger_bands_width_percentage(warm_candles, bb_period)
        kcp = calculate_keltner_channel_percentage(warm_candles, kc_period)
        
        # If any indicator calculation results in NaN (due to insufficient data or edge cases), skip.
        if np.isnan(rsi) or np.isnan(bbw) or np.isnan(kcp):
            continue

        # Calculate Bid-Ask Spread Percentage from the latest tick.
        if latest_tick.last_price == 0: # Avoid division by zero.
            continue
        bid_ask_spread_pct = (latest_tick.ask_price - latest_tick.bid_price) / latest_tick.last_price * 100

        # Apply the rule conditions for a BUY signal.
        if (70 <= rsi <= 88 and
            5 <= bbw <= 14 and
            kcp > 180 and
            bid_ask_spread_pct < 0.5):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_tick.polled_at,
                price=latest_tick.last_price,
                rule_id=rule_id,
                confidence=None, # No specific confidence logic provided in the idea.
                indicators={
                    "rsi": rsi,
                    "bbw": bbw,
                    "kcp": kcp,
                    "bid_ask_spread_pct": bid_ask_spread_pct
                }
            ))

    return signals