from __future__ import annotations
import statistics
import numpy as np
from scipy.stats import f
from src.agent.models import BuySignal, SellSignal, PairData

# --- Rule Configuration ---
RULE_ID = "rule_12_lead_lag_v2"

# Parameters for Granger Causality Test
CAUSALITY_LOOKBACK_WINDOW = 50  # Number of data points (returns) for the Granger causality test
MAX_LAG_TO_TEST = 5             # Maximum lag period to test for Granger causality
GRANGER_SIGNIFICANCE_LEVEL = 0.05 # p-value threshold for statistical significance

# Parameters for Adaptive Thresholds
VOLATILITY_WINDOW = 20          # Window (returns) for calculating asset A's volatility
VOLATILITY_MULTIPLIER = 1.5     # Multiplier for volatility to set buy/sell thresholds

# Minimum number of price ticks required for both leading and lagging assets.
# This ensures enough data for return calculations and all lookback windows.
# The calculation considers:
# 1. Log returns reduce the data length by 1.
# 2. Causality window needs `CAUSALITY_LOOKBACK_WINDOW` returns.
# 3. Volatility window needs `VOLATILITY_WINDOW` returns.
# 4. `current_time_point - optimal_lag` needs valid index for `asset_A_returns`.
# Therefore, `len(prices)` must be at least `max(CAUSALITY_LOOKBACK_WINDOW, VOLATILITY_WINDOW, MAX_LAG_TO_TEST) + 2`.
# (+2 accounts for the latest return index, and the fact that returns are 1 less than prices).
MIN_TICKS = max(CAUSALITY_LOOKBACK_WINDOW, VOLATILITY_WINDOW, MAX_LAG_TO_TEST) + 2

# Define leading-lagging asset pairs.
# Key: Lagging asset symbol (B), Value: Leading asset symbol (A)
# This mapping needs to be defined based on market analysis or configuration.
LEADING_ASSET_MAP = {
    "ETH/USD": "BTC/USD",
    # Add other pairs as needed, e.g.,
    # "LTC/USD": "BTC/USD",
    # "SOL/USD": "ETH/USD",
}

MarketData = dict[str, PairData]

# --- Helper Functions ---

def _calculate_log_returns(prices: list[float]) -> np.ndarray:
    """Calculates log returns from a list of prices."""
    if len(prices) < 2:
        return np.array([])
    return np.diff(np.log(prices))

def _granger_causality_test(
    data_x: np.ndarray, data_y: np.ndarray, lag: int
) -> float:
    """
    Performs a simplified Granger causality test using F-statistic on OLS residuals
    and returns the p-value. This implementation avoids `statsmodels` by using `numpy`
    for OLS and `scipy.stats.f` for the F-distribution.

    It compares the Sum of Squared Residuals (SSR) of a restricted model (Y explained by past Y)
    against an unrestricted model (Y explained by past Y and past X).
    The F-statistic tests if the lagged X variables significantly improve the prediction of Y.
    """
    data_x = np.asarray(data_x).flatten()
    data_y = np.asarray(data_y).flatten()

    # Number of observations for the regression (after accounting for lags)
    # The dependent variable Y_t will start at index `lag`
    n_obs = len(data_y) - lag
    if n_obs <= 0: # Not enough observations for the given lag
        return 1.0

    # Dependent variable Y_t
    y_target = data_y[lag:]

    # --- Restricted Model: Y_t = c + a_1*Y_{t-1} + ... + a_lag*Y_{t-lag} + e_t ---
    # Design matrix for restricted model: [constant, Y_{t-1}, ..., Y_{t-lag}]
    restricted_design_matrix = np.ones((n_obs, lag + 1)) # +1 for constant term
    for i in range(lag):
        # Y_{t-(i+1)}
        restricted_design_matrix[:, i+1] = data_y[lag - (i+1) : n_obs + lag - (i+1)]

    try:
        # Perform OLS regression
        _, residuals_restricted_array, _, _ = np.linalg.lstsq(restricted_design_matrix, y_target, rcond=None)
        if residuals_restricted_array.size > 0:
            ssr_restricted = residuals_restricted_array[0]
        else:
            # If residuals_array is empty, it typically means a perfect fit or
            # number of observations equals number of parameters. Assume SSR is 0.
            ssr_restricted = 0.0
    except np.linalg.LinAlgError:
        return 1.0 # Regression failed, return no causality (high p-value)

    # --- Unrestricted Model: Y_t = c + a_1*Y_{t-1} + ... + a_lag*Y_{t-lag} + b_1*X_{t-1} + ... + b_lag*X_{t-lag} + u_t ---
    # Design matrix for unrestricted model: [constant, Y_{t-1}..Y_{t-lag}, X_{t-1}..X_{t-lag}]
    x_lagged_matrix = np.zeros((n_obs, lag))
    for i in range(lag):
        # X_{t-(i+1)}
        x_lagged_matrix[:, i] = data_x[lag - (i+1) : n_obs + lag - (i+1)]

    unrestricted_design_matrix = np.hstack([restricted_design_matrix, x_lagged_matrix])

    try:
        _, residuals_unrestricted_array, _, _ = np.linalg.lstsq(unrestricted_design_matrix, y_target, rcond=None)
        if residuals_unrestricted_array.size > 0:
            ssr_unrestricted = residuals_unrestricted_array[0]
        else:
            ssr_unrestricted = 0.0
    except np.linalg.LinAlgError:
        return 1.0 # Regression failed, return no causality (high p-value)

    # --- F-statistic Calculation ---
    # Number of restrictions (coefficients of lagged X that are tested to be zero)
    df1 = lag
    # Number of parameters in the unrestricted model (intercept + lag Y + lag X)
    k = 1 + lag + lag
    df2 = n_obs - k

    # Ensure degrees of freedom are positive for the F-test
    if df1 <= 0 or df2 <= 0 or ssr_unrestricted < 1e-10:
        # If ssr_unrestricted is near zero, it implies a very good fit.
        # If ssr_unrestricted is truly zero and ssr_restricted is not, it means perfect prediction
        # with X, suggesting strong causality. If both are zero, it's ambiguous.
        if ssr_unrestricted < 1e-10 and ssr_restricted > 1e-10:
            return 0.0 # strong causality
        return 1.0 # Not enough data for a valid F-test or no improvement

    # F-statistic formula: F = ((SSR_R - SSR_U) / df1) / (SSR_U / df2)
    numerator = (ssr_restricted - ssr_unrestricted) / df1
    denominator = ssr_unrestricted / df2

    if denominator <= 0: # Defensive check, should not happen if ssr_unrestricted > 1e-10
        if numerator > 0: return 0.0 # Strong evidence of causality if numerator positive
        return 1.0 # No evidence

    f_statistic = numerator / denominator

    # Get the p-value from the F-distribution using the survival function (1 - CDF)
    p_value = f.sf(f_statistic, df1, df2)

    return p_value

def lead_lag_v2(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates buy/sell signals for lagging assets based on Granger causality
    and adaptive volatility thresholds of leading assets.
    """
    signals: list[BuySignal | SellSignal] = []

    for lagging_asset_symbol, pair_data_b in data.items():
        if lagging_asset_symbol not in LEADING_ASSET_MAP:
            continue # This asset is not configured as a lagging asset in any pair

        leading_asset_symbol = LEADING_ASSET_MAP[lagging_asset_symbol]

        if leading_asset_symbol not in data:
            continue # Leading asset data not available in the provided MarketData

        pair_data_a = data[leading_asset_symbol]

        # Extract prices for leading (A) and lagging (B) assets
        asset_A_prices = [tick.last_price for tick in pair_data_a.hot]
        asset_B_prices = [tick.last_price for tick in pair_data_b.hot]

        # Ensure we have enough price data for all lookback windows and return calculations
        if len(asset_A_prices) < MIN_TICKS or len(asset_B_prices) < MIN_TICKS:
            continue

        # Calculate log returns
        asset_A_returns = _calculate_log_returns(asset_A_prices)
        asset_B_returns = _calculate_log_returns(asset_B_prices)

        # `current_time_point` refers to the index of the latest return in `asset_B_returns`.
        # A signal is generated for the price corresponding to `asset_B_returns[current_time_point]`.
        # This corresponds to `pair_data_b.hot[current_time_point + 1]`.
        current_time_point = len(asset_B_returns) - 1

        # Check if there are enough returns for all calculations required by `current_time_point`.
        # The earliest index needed for any calculation (causality, volatility, lagged movement)
        # for `current_time_point` must be non-negative.
        min_required_return_idx = max(CAUSALITY_LOOKBACK_WINDOW, VOLATILITY_WINDOW, MAX_LAG_TO_TEST) - 1
        if current_time_point < min_required_return_idx:
            continue # Not enough historical returns to perform calculations for the latest point

        # --- Granger Causality Test ---
        # Data for causality test: `CAUSALITY_LOOKBACK_WINDOW` returns ending at `current_time_point - 1`.
        # The pseudocode's `get_historical_data(..., current_time_point - 1)` implies
        # a slice `[start_idx : current_time_point]` to get data up to index `current_time_point - 1`.
        causality_start_idx = current_time_point - CAUSALITY_LOOKBACK_WINDOW
        causality_end_idx = current_time_point # Exclusive upper bound
        
        causality_data_A = asset_A_returns[causality_start_idx : causality_end_idx]
        causality_data_B = asset_B_returns[causality_start_idx : causality_end_idx]

        optimal_lag: int | None = None
        min_p_value = 1.0

        for lag in range(1, MAX_LAG_TO_TEST + 1):
            p_value = _granger_causality_test(causality_data_A, causality_data_B, lag)

            if p_value < min_p_value:
                min_p_value = p_value
                optimal_lag = lag
            
            # Optimization: If we found a perfect p-value (0.0), no need to test further lags.
            if min_p_value == 0.0: