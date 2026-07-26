from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle, Tick


def calculate_sma(prices: list[float], period: int) -> float:
    """
    Calculates the Simple Moving Average (SMA) for the last `period` prices.
    """
    if len(prices) < period:
        return 0.0  # Not enough data to calculate SMA
    return sum(prices[-period:]) / period


def calculate_rsi(prices: list[float], period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for the last price in the series.
    Uses Wilder's smoothing method.
    """
    if len(prices) < period + 1:
        # Not enough data to calculate even the first RSI value
        return 0.0  # Return 0.0 if data is insufficient

    price_arr = np.array(prices, dtype=np.float64)

    # Calculate price changes
    deltas = price_arr[1:] - price_arr[:-1]

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, abs(deltas), 0)

    # Calculate initial average gain and average loss (SMA for the first 'period' values)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Apply Wilder's smoothing method (EMA) for subsequent periods
    # The loop starts from `period` because the initial averages cover `gains[0]` to `gains[period-1]`.
    # `gains[period]` is the first delta for which we apply smoothing.
    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    # Calculate Relative Strength (RS)
    if avg_loss == 0:
        # If there are no losses, RSI is 100.
        # If there are also no gains (avg_gain == 0), then the price is flat, RSI is 50.
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Overbought RSI False Breakout Short' trading strategy.

    This strategy identifies potential shorting opportunities when a seemingly bullish
    price crossover above a 10-period Simple Moving Average (SMA) occurs in conjunction
    with an overbought Hourly Relative Strength Index (RSI > 60). It exploits the
    observed tendency for such 'bullish' signals to fail and lead to losses in
    overextended market conditions.
    """
    signals: list[BuySignal | SellSignal] = []
    
    for pair, pair_data in data.items():
        warm_candles = pair_data.warm
        hot_ticks = pair_data.hot

        # Ensure we have enough warm candle data for SMA and RSI calculations
        # RSI(14) needs at least 14+1 = 15 candles.
        # SMA(10) crossover check needs 10+1 = 11 candles.
        # So, 15 is the minimum required.
        if len(warm_candles) < 15:
            continue

        # Ensure there's at least one hot tick for the entry price
        if not hot_ticks:
            continue

        hourly_closes = [c.close for c in warm_candles]

        # --- Calculate SMA for crossover check ---
        sma_period = 10
        # The SMA is calculated on the `sma_period` candles immediately preceding
        # the last two completed candles.
        # hourly_closes[-(sma_period + 1):-1] slices the list to get `sma_period`
        # candles ending before the second-to-last candle.
        sma_for_crossover_check = calculate_sma(hourly_closes[-(sma_period + 1):-1], sma_period)
        
        # --- Calculate current Hourly RSI ---
        rsi_period = 14
        current_hourly_rsi = calculate_rsi(hourly_closes, rsi_period)

        # Retrieve closing prices for the last two completed candles
        previous_candle_close = hourly_closes[-2]
        last_completed_candle_close = hourly_closes[-1]

        # --- Check for crossover above SMA ---
        # A bullish crossover occurs if the previous candle closed below the SMA
        # and the last completed candle closed above the SMA.
        crossover_above_sma_detected = (
            previous_candle_close < sma_for_crossover_check and
            last_completed_candle_close > sma_for_crossover_check
        )

        # --- Check for RSI overbought condition ---
        rsi_overbought = current_hourly_rsi > 60

        # --- Generate SHORT signal if both conditions are met ---
        if crossover_above_sma_detected and rsi_overbought:
            signals.append(SellSignal(
                pair=pair,
                timestamp=hot_ticks[-1].polled_at,
                price=hot_ticks[-1].last_price,
                rule_id="OVRSI-FBB-SHORT-001",
            ))
            
    return signals