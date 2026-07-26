from __future__ import annotations
import numpy as np
from src.agent.models import BuySignal, MarketData, SellSignal
from datetime import datetime

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

    if avg_gain == 0 and avg_loss == 0:
        return 50.0 # Neutral, no price change in the period
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
        # If sum_abs_changes is also 0, it means price was flat, PRR would be 1.0.
        return 1.0 if sum_abs_changes == 0 else float(np.inf)
    
    return float(sum_abs_changes / total_change)

def _calculate_bollinger_band_width(prices: list[float], period: int) -> float:
    """
    Calculates the Bollinger Band Width (BBW).
    BBW = (4 * Standard Deviation) / SMA. (Assuming 2-sigma bands, so 2*2=4 for total width)
    Requires at least 'period' prices.
    """
    if len(prices) < period:
        raise ValueError(f"Not enough data for BBW calculation. Needed: {period}, Got: {len(prices)}")

    relevant_prices = np.array(prices[-period:])
    
    sma = np.mean(relevant_prices)
    
    # Use sample standard deviation (ddof=1) for consistency with typical indicator calculations
    # If period is 1, std dev is 0.
    std_dev = np.std(relevant_prices, ddof=1) if period > 1 else 0.0

    if sma == 0: # Avoid division by zero if SMA is zero
        return 0.0

    # BBW is typically (Upper Band - Lower Band) / Middle Band (SMA).
    # Upper = SMA + 2*STDDEV, Lower = SMA - 2*STDDEV
    # So (SMA + 2*STDDEV) - (SMA - 2*STDDEV) = 4*STDDEV
    # BBW = (4 * STDDEV) / SMA
    bbw = (4 * std_dev) / sma
    return float(bbw)

# --- Main signal generation function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the 'Refined 10-Period SMA Crossover with Volatility and Momentum Filters' trading rule.
    Triggers BUY signals based on SMA crossover, filtered by RSI, Bid-Ask Spread,
    Price Reversion Ratio, and Bollinger Band Width.
    """
    signals: list[BuySignal | SellSignal] = []
    
    # Define required lookback periods for the rule's indicators
    SMA_PERIOD = 10
    RSI_PERIOD = 14
    PRR_PERIOD = 20
    BBW_PERIOD = 20

    # Determine the minimum number of hourly candles needed for all calculations
    # For SMA crossover:
    #   _calculate_sma(hourly_closes, SMA_PERIOD) needs len(hourly_closes) >= SMA_PERIOD
    #   _calculate_sma(hourly_closes[:-1], SMA_PERIOD) needs len(hourly_closes) - 1 >= SMA_PERIOD, so len(hourly_closes) >= SMA_PERIOD + 1
    # RSI and PRR require (PERIOD + 1) candles. BBW requires PERIOD candles.
    MIN_CANDLES_REQUIRED = max(SMA_PERIOD + 1, RSI_PERIOD + 1, PRR_PERIOD + 1, BBW_PERIOD)

    for pair, pair_data in data.items():
        # Extract hourly close prices from warm data
        hourly_closes = [c.close for c in pair_data.warm]
        
        # Ensure enough warm data for all indicator calculations and crossover logic
        # Also, hourly_closes[-2] requires at least 2 candles. This is covered by MIN_CANDLES_REQUIRED >= SMA_PERIOD + 1 >= 11
        if len(hourly_closes) < MIN_CANDLES_REQUIRED:
            continue

        # Ensure hot data is available for signal timestamp, price, and Bid-Ask Spread Percentage
        if not pair_data.hot:
            continue
        
        # Get the most recent hot tick for current price and BASP
        latest_tick = pair_data.hot[-1]

        try:
            # Calculate SMAs for crossover detection
            # sma_10: SMA calculated using all available completed hourly closes (hourly_closes)
            current_sma_10 = _calculate_sma(hourly_closes, SMA_PERIOD)
            # sma_10_previous_candle: SMA calculated using all completed hourly closes *except* the very last one (hourly_closes[:-1])
            prev_sma_10_for_crossover = _calculate_sma(hourly_closes[:-1], SMA_PERIOD) 

            # Calculate filter indicators using the latest relevant data
            rsi_value = _calculate_rsi(hourly_closes, RSI_PERIOD)
            prr_value = _calculate_price_reversion_ratio(hourly_closes, PRR_PERIOD)
            bbw_value = _calculate_bollinger_band_width(hourly_closes, BBW_PERIOD)

            # Get the Bid-Ask Spread Percentage from the most recent hot tick
            # spread_rel is already a percentage (e.g., 0.5 for 0.5%)
            hot_basp = latest_tick.spread_rel

            # Apply filter conditions based on predefined thresholds
            is_rsi_moderate = 30 <= rsi_value <= 60
            is_bbw_sufficient = 0.0005 <= bbw_value <= 90
            is_prr_healthy = 0 < prr_value < 40 # prr_value can be inf, which fails this filter.
            is_basp_low = hot_basp < 1.0 # < 1.0% spread

            # Get current price from hot data and previous candle close for crossover logic
            current_price = latest_tick.last_price
            prev_close_candle = hourly_closes[-2] # Close of the second-to-last completed hourly candle

            # Check for BUY signal conditions:
            # 1. Current live price (from hot data) is above the SMA calculated on all completed warm candles.
            # 2. The close of the second-to-last completed warm candle was below or equal to the SMA calculated
            #    on all warm candles *except* the last one.
            if current_price > current_sma_10 and prev_close_candle <= prev_sma_10_for_crossover:
                if is_rsi_moderate and is_bbw_sufficient and is_prr_healthy and is_basp_low:
                    signals.append(BuySignal(
                        pair=pair,
                        timestamp=latest_tick.polled_at,
                        price=latest_tick.last_price,
                        rule_id="fix_rule01_filters_v2", # Updated rule ID as per idea_id
                        indicators={
                            "sma_10": current_sma_10,
                            "rsi_14": rsi_value,
                            "prr_20": prr_value, # Updated period in indicator name
                            "bbw_20": bbw_value,
                            "basp": hot_basp,
                        }
                    ))
            # The rule description and pseudocode only specify a BUY signal.
            # No SELL signal logic is implemented for this specific rule.
            
        except (ValueError, IndexError, ZeroDivisionError):
            # Catch potential errors during indicator calculation (e.g., insufficient data not caught
            # by initial length checks in specific indicator functions, or numerical issues).
            # Continue to the next pair if an error occurs for the current one.
            continue
            
    return signals