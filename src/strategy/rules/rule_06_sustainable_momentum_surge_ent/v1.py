from __future__ import annotations
import statistics
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick
from datetime import datetime

# Helper functions for technical indicators

def calculate_sma(data: list[float], N: int) -> float | None:
    """Calculates Simple Moving Average."""
    if len(data) < N:
        return None
    return sum(data[-N:]) / N

def calculate_stddev(data: list[float], N: int) -> float | None:
    """Calculates Standard Deviation."""
    if len(data) < N:
        return None
    # statistics.stdev requires at least 2 data points.
    if N < 2:
        if N == 1: 
            return 0.0  # Standard deviation of a single data point is 0
        return None
    return statistics.stdev(data[-N:])

def calculate_rsi(closes: list[float], N: int) -> float | None:
    """Calculates Relative Strength Index."""
    # Need N+1 closes to get N price changes for the initial average
    if len(closes) < N + 1:
        return None

    # Calculate price differences (changes)
    diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    # Use the last N differences for the RSI calculation
    gains_last_N = [d if d > 0 else 0 for d in diffs[-N:]]
    losses_last_N = [abs(d) if d < 0 else 0 for d in diffs[-N:]]

    avg_gain_last_N = sum(gains_last_N) / N
    avg_loss_last_N = sum(losses_last_N) / N

    if avg_loss_last_N == 0:
        return 100.0 if avg_gain_last_N > 0 else 50.0 # If no losses, RSI is 100. If no gains either, it's neutral.
    
    rs = avg_gain_last_N / avg_loss_last_N
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs: list[float], lows: list[float], closes: list[float], N: int) -> float | None:
    """Calculates Average True Range."""
    # Need N+1 bars to calculate N True Ranges for the initial average
    if len(highs) < N + 1 or len(lows) < N + 1 or len(closes) < N + 1:
        return None
    
    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        true_ranges.append(tr)
    
    # Ensure we have at least N true_ranges for the average
    if len(true_ranges) < N:
        return None

    # Calculate ATR using the last N true ranges
    atr = sum(true_ranges[-N:]) / N
    return atr

def calculate_kcp(highs: list[float], lows: list[float], closes: list[float], N: int) -> float | None:
    """Calculates Keltner Channel Percentage (price position relative to Keltner Channel)."""
    # KCP needs SMA(N) and ATR(N). SMA needs N closes, ATR needs N+1 closes.
    # Therefore, N+1 closes are required overall.
    if len(closes) < N + 1:
        return None
    
    sma_val = calculate_sma(closes, N)
    atr_val = calculate_atr(highs, lows, closes, N)

    if sma_val is None or atr_val is None:
        return None

    # Keltner Channel multiplier is typically 2.
    multiplier = 2 
    middle_band = sma_val
    upper_band = middle_band + multiplier * atr_val
    lower_band = middle_band - multiplier * atr_val

    if upper_band == lower_band: # Avoid division by zero in case of zero ATR or flat market
        return 0.0 
    
    latest_close = closes[-1]
    kcp = ((latest_close - lower_band) / (upper_band - lower_band)) * 100
    return kcp

def calculate_bbw(closes: list[float], N: int) -> float | None:
    """Calculates Bollinger Band Width."""
    if len(closes) < N:
        return None
    
    sma_val = calculate_sma(closes, N)
    stddev_val = calculate_stddev(closes, N)

    if sma_val is None or stddev_val is None:
        return None
    
    # Standard deviation multiplier for Bollinger Bands is typically 2.
    multiplier = 2
    
    # BBW = (Upper Band - Lower Band) / Middle Band * 100
    # Upper Band = SMA + Multiplier * StdDev
    # Lower Band = SMA - Multiplier * StdDev
    # (Upper - Lower) = 2 * Multiplier * StdDev
    # BBW = (2 * Multiplier * StdDev) / SMA * 100
    
    if sma_val == 0: # Avoid division by zero
        return 0.0 

    bbw = (2 * multiplier * stddev_val) / sma_val * 100
    return bbw


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []
    rule_id = "momentum_surge_entry"

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Initial data checks from pseudocode
        if len(warm_candles) < 20 or not hot_ticks:
            continue
        
        # Extract OHLC data from warm candles
        closes = [c.close for c in warm_candles]
        highs = [c.high for c in warm_candles]
        lows = [c.low for c in warm_candles]

        # Consolidated check for minimum data required by all indicators:
        # RSI(14) needs at least 14+1 = 15 closes.
        # KCP(20) needs ATR(20), which needs 20+1 = 21 closes.
        # BBW(20) needs at least 20 closes.
        # Therefore, we need a minimum of 21 closes to calculate all indicators properly.
        if len(closes) < 21:
            continue

        # Calculate indicators
        rsi_val = calculate_rsi(closes, 14)
        kcp_val = calculate_kcp(highs, lows, closes, 20)
        bbw_val = calculate_bbw(closes, 20)

        # If any indicator calculation failed (e.g., due to edge cases in data), skip
        if rsi_val is None or kcp_val is None or bbw_val is None:
            continue

        # Get latest tick data for spread and price
        latest_tick = hot_ticks[-1]
        bid_price = latest_tick.bid_price
        ask_price = latest_tick.ask_price

        # Ensure valid bid/ask prices for spread calculation
        if bid_price <= 0 or ask_price <= 0:
            continue
        
        # Calculate Bid-Ask Spread Percentage
        spread_percentage = ((ask_price - bid_price) / ask_price) * 100

        # Apply the trading rule conditions
        if (70 <= rsi_val <= 85 and
            150 <= kcp_val <= 220 and
            8 <= bbw_val <= 25 and
            1.0 <= spread_percentage <= 7.0):
            
            signals.append(BuySignal(
                pair=pair,
                timestamp=latest_tick.polled_at,
                price=latest_tick.last_price,
                rule_id=rule_id,
                reason="Strong momentum surge with healthy volatility and liquidity."
            ))
            
    return signals