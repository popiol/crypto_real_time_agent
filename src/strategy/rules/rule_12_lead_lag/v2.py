from __future__ import annotations
import statistics
import numpy as np
from src.agent.models import BuySignal, SellSignal, PairData

RULE_ID = "rule_12_lead_lag_v2"

# --- Rule Parameters ---
# These parameters are hardcoded as per the pseudocode's example.
# In a real system, these would typically be configurable per trading pair or strategy instance.
LEAD_ASSET_ID = "BTC"  # Example leading asset (e.g., Bitcoin)
LAG_ASSET_ID = "ETH"   # Example lagging asset (e.g., Ethereum)
LAG_PERIOD = 10        # Number of historical ticks to look back for the leading asset's significant move
LEAD_RETURN_THRESHOLD_SIGMA = 1.5 # Multiplier for leading asset's volatility to define threshold
LAG_ASSET_TREND_FAST_EMA_PERIOD = 10 # Period for the fast EMA on the lagging asset
LAG_ASSET_TREND_SLOW_EMA_PERIOD = 20 # Period for the slow EMA on the lagging asset
# --- End Rule Parameters ---

MarketData = dict[str, PairData]

def _calculate_ema(prices: np.ndarray, period: int) -> float:
    """
    Calculates the Exponential Moving Average (EMA) for a given series of prices.
    Assumes prices are in chronological order (oldest to newest).
    """
    if len(prices) < period:
        # Not enough data for EMA calculation
        return np.nan

    alpha = 2 / (period + 1)
    # Initialize EMA with the Simple Moving Average (SMA) of the first 'period' prices
    ema = np.mean(prices[:period])

    # Calculate subsequent EMAs
    for price in prices[period:]:
        ema = (price * alpha) + (ema * (1 - alpha))
    return ema

def _calculate_volatility(returns: np.ndarray) -> float:
    """
    Calculates the standard deviation (volatility) of a series of returns.
    Returns 0.0 if there are fewer than 2 returns to avoid errors.
    """
    if len(returns) < 2:
        return 0.0
    return np.std(returns)

def lead_lag_v2(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Implements the "Volatility-Adjusted Lead-Lag with Trend Confirmation" trading rule.

    Detects significant price movements in a leading asset, but only emits a signal
    for the lagging asset if the leading asset's return exceeds a volatility-adjusted
    threshold AND the lagging asset's short-term trend confirms the direction.
    """
    signals: list[BuySignal | SellSignal] = []

    # Ensure both leading and lagging assets are present in the provided market data
    if LEAD_ASSET_ID not in data or LAG_ASSET_ID not in data:
        return []

    lead_asset_pair_data = data[LEAD_ASSET_ID]
    lag_asset_pair_data = data[LAG_ASSET_ID]

    lead_ticks = lead_asset_pair_data.hot
    lag_ticks = lag_asset_pair_data.hot

    # Determine the minimum number of ticks required for all calculations:
    # 1. Leading asset's lagged return: needs current price and price LAG_PERIOD ticks ago, so LAG_PERIOD + 1 ticks.
    # 2. Leading asset's volatility: needs LAG_PERIOD returns, which means LAG_PERIOD + 1 ticks.
    # 3. Lagging asset's EMAs: needs at least LAG_ASSET_TREND_SLOW_EMA_PERIOD ticks for the slowest EMA.
    min_ticks_required = max(LAG_PERIOD + 1, LAG_ASSET_TREND_SLOW_EMA_PERIOD)

    # Check for sufficient historical data for both assets
    if len(lead_ticks) < min_ticks_required or len(lag_ticks) < min_ticks_required:
        return []

    # --- Leading Asset Calculations ---
    lead_prices = np.array([t.last_price for t in lead_ticks])
    current_price_lead = lead_prices[-1]
    # Price of the leading asset LAG_PERIOD ticks ago
    price_lead_lagged = lead_prices[-1 - LAG_PERIOD]

    # Calculate the leading asset's return over the LAG_PERIOD
    lead_asset_lagged_return = (current_price_lead - price_lead_lagged) / price_lead_lagged

    # Calculate the leading asset's recent volatility
    # Use prices covering the last LAG_PERIOD returns for volatility calculation
    volatility_prices_lead = lead_prices[-LAG_PERIOD-1:]
    lead_returns = np.diff(volatility_prices_lead) / volatility_prices_lead[:-1]
    volatility_lead = _calculate_volatility(lead_returns)

    # Calculate volatility-adjusted thresholds
    adjusted_buy_threshold = LEAD_RETURN_THRESHOLD_SIGMA * volatility_lead
    adjusted_sell_threshold = -LEAD_RETURN_THRESHOLD_SIGMA * volatility_lead

    # --- Lagging Asset Calculations ---
    lag_prices = np.array([t.last_price for t in lag_ticks])

    # Calculate EMAs for the lagging asset
    fast_ema_lag = _calculate_ema(lag_prices, LAG_ASSET_TREND_FAST_EMA_PERIOD)
    slow_ema_lag = _calculate_ema(lag_prices, LAG_ASSET_TREND_SLOW_EMA_PERIOD)

    # If EMA calculation resulted in NaN (e.g., due to insufficient data, though
    # min_ticks_required should prevent this if periods are reasonable), skip signal generation.
    if np.isnan(fast_ema_lag) or np.isnan(slow_ema_lag):
        return []

    # --- Signal Generation ---
    current_lag_asset_price = lag_ticks[-1].last_price
    current_lag_asset_timestamp = lag_ticks[-1].polled_at

    # Check for Buy signal
    if lead_asset_lagged_return > adjusted_buy_threshold:
        # Confirm bullish trend in lagging asset
        if fast_ema_lag > slow_ema_lag:
            signals.append(BuySignal(
                pair=LAG_ASSET_ID,
                rule_id=RULE_ID,
                timestamp=current_lag_asset_timestamp,
                price=current_lag_asset_price,
            ))

    # Check for Sell signal
    if lead_asset_lagged_return < adjusted_sell_threshold:
        # Confirm bearish trend in lagging asset
        if fast_ema_lag < slow_ema_lag:
            signals.append(SellSignal(
                pair=LAG_ASSET_ID,
                rule_id=RULE_ID,
                timestamp=current_lag_asset_timestamp,
                price=current_lag_asset_price,
            ))

    return signals