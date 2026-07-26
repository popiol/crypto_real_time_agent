from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal

# --- Helper functions for indicator calculations ---
# These functions calculate the latest value of each indicator based on a list of prices.

def _calculate_sma(prices: list[float], period: int) -> float:
    """
    Calculates the Simple Moving Average (SMA) for the given prices.
    Requires at least 'period' prices.
    """
    if len(prices) < period:
        raise ValueError(f"Not enough data for SMA calculation. Needed: {period}, Got: {len(prices)}")
    return float(np.mean(prices[-period:]))

def _calculate_rsi(prices: list[float], period: int) -> float:
    """
    Calculates the Relative Strength Index (RSI) for the given prices.
    Requires at least (period + 1) prices to derive 'period' price changes.
    """
    if len(prices) <= period:
        raise ValueError(f"Not enough data for RSI calculation. Needed: {period + 1}, Got: {len(prices)}")

    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0) # Convert losses to positive values

    # Calculate average gain/loss over the most recent 'period' changes
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0 # No losses, RSI is 100
    if avg_gain == 0:
        return 0.0 # No gains, RSI is 0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def _calculate_price_reversion_ratio(prices: list[float], period: int) -> float:
    """
    Calculates a Price Reversion Ratio (PRR) as the sum of absolute price changes
    divided by the absolute total change over the period.
    A value closer to 1.0 indicates a relatively straight trend (less choppiness).
    Requires at least (period + 1) prices.
    """
    if len(prices) <= period:
        raise ValueError(f"Not enough data for PRR calculation. Needed: {period + 1}, Got: {len(prices)}")

    relevant_prices = np.array(prices[-(period + 1):])
    
    sum_abs_changes = np.sum(np.abs(np.diff(relevant_prices)))
    total_change = np.abs(relevant_prices[-1] - relevant_prices[0])

    if total_change == 0:
        # If there's no net change but sum_abs_changes > 0, it indicates high choppiness
        # where price returned to its starting point. Return infinity for this case.
        return 1.0 if sum_abs_changes == 0 else float(np.inf)
    
    return float(sum_abs_changes / total_change)

def _calculate_bollinger_band_width(prices: list[float], period: int) -> float:
    """
    Calculates the Bollinger Band Width (BBW).
    BBW = (4 * Standard Deviation) / SMA.
    Requires at least 'period' prices.
    """
    if len(prices) < period:
        raise ValueError(f"Not enough data for BBW calculation. Needed: {period}, Got: {len(prices)}")

    relevant_prices = np.array(prices[-period:])
    
    sma = np.mean(relevant_prices)
    
    # Use sample standard deviation (ddof=1) for consistency with typical indicator calculations
    if period > 1:
        std_dev = np.std(relevant_prices, ddof=1)
    else: # If period is 1, standard deviation is 0
        std_dev = 0.0

    if sma == 0: # Avoid division by zero if SMA is zero
        return 0.0

    bbw = (4 * std_dev) / sma
    return float(bbw)

# --- Main signal generation function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Filtered Hourly SMA(20) Crossover for Trend Confirmation' trading rule.
    Triggers long/short signals based on SMA crossover, filtered by RSI, Bid-Ask Spread,
    Price Reversion Ratio, and Bollinger Band Width.
    """
    signals: list[BuySignal | SellSignal] = []
    
    # Define required lookback periods for the rule's indicators
    SMA_PERIOD = 20
    RSI_PERIOD = 14
    PRR_PERIOD = 14
    BBW_PERIOD = 20

    # Determine the minimum number of hourly candles needed for all calculations
    # SMA crossover requires (SMA_PERIOD + 1) candles to compare current and previous SMA.
    # RSI and PRR require (PERIOD + 1) candles. BBW requires PERIOD candles.
    MIN_CANDLES_REQUIRED = max(SMA_PERIOD + 1, RSI_PERIOD + 1, PRR_PERIOD + 1, BBW_PERIOD)

    for pair, pair_data in data.items():
        # Extract hourly close prices from warm data
        hourly_closes = [c.close for c in pair_data.warm]
        
        # Ensure enough warm data for all indicator calculations and crossover logic
        if len(hourly_closes) < MIN_CANDLES_REQUIRED:
            continue

        # Ensure hot data is available for signal timestamp, price, and Bid-Ask Spread Percentage
        if not pair_data.hot:
            continue

        try:
            # Calculate current and previous SMA for crossover detection
            current_sma_20 = _calculate_sma(hourly_closes, SMA_PERIOD)
            prev_sma_20 = _calculate_sma(hourly_closes[:-1], SMA_PERIOD) # SMA ending one candle ago

            # Calculate filter indicators using the latest relevant data
            rsi_14 = _calculate_rsi(hourly_closes, RSI_PERIOD)
            prr_14 = _calculate_price_reversion_ratio(hourly_closes, PRR_PERIOD)
            bbw_20 = _calculate_bollinger_band_width(hourly_closes, BBW_PERIOD)

            # Get the Bid-Ask Spread Percentage from the most recent hot tick
            hot_basp = pair_data.hot[-1].spread_rel

            # Apply filter conditions based on predefined thresholds
            is_rsi_moderate = 42 <= rsi_14 <= 52
            is_basp_low = hot_basp < 0.3
            is_prr_healthy = 1.0 <= prr_14 <= 5.5
            is_bbw_sufficient = bbw_20 > 0.0001

            # Get current and previous close prices for crossover logic
            current_close = hourly_closes[-1]
            prev_close = hourly_closes[-2]

            # Check for BUY signal conditions: price crosses above SMA and all filters pass
            if current_close > current_sma_20 and prev_close <= prev_sma_20:
                if is_rsi_moderate and is_basp_low and is_prr_healthy and is_bbw_sufficient:
                    signals.append(BuySignal(
                        pair=pair,
                        timestamp=pair_data.hot[-1].polled_at,
                        price=pair_data.hot[-1].last_price,
                        rule_id="filtered_hourly_sma20_crossover",
                        indicators={
                            "sma_20": current_sma_20,
                            "rsi_14": rsi_14,
                            "prr_14": prr_14,
                            "bbw_20": bbw_20,
                            "basp": hot_basp,
                        }
                    ))
            # Check for SELL signal conditions: price crosses below SMA and all filters pass
            elif current_close < current_sma_20 and prev_close >= prev_sma_20:
                if is_rsi_moderate and is_basp_low and is_prr_healthy and is_bbw_sufficient:
                    signals.append(SellSignal(
                        pair=pair,
                        timestamp=pair_data.hot[-1].polled_at,
                        price=pair_data.hot[-1].last_price,
                        rule_id="filtered_hourly_sma20_crossover",
                        indicators={
                            "sma_20": current_sma_20,
                            "rsi_14": rsi_14,
                            "prr_14": prr_14,
                            "bbw_20": bbw_20,
                            "basp": hot_basp,
                        }
                    ))
        except (ValueError, IndexError):
            # Catch potential errors during indicator calculation (e.g., specific edge cases
            # not fully covered by initial length checks, or numerical issues).
            # Continue to the next pair if an error occurs for the current one.
            continue
            
    return signals