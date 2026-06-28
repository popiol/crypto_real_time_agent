"""Rule 03 — Stochastic process: Ornstein-Uhlenbeck spread mean reversion.

Fits an OU process to the hot-tier spread time series to estimate the
long-run equilibrium spread (μ) and residual volatility (σ).

Buy signal:  spread significantly above μ AND compressing (slope < 0)
             → market becoming liquid after illiquidity, price move likely.
Sell signal: spread significantly below μ AND expanding (slope > 0)
             → liquidity deteriorating, exit condition.
"""

from __future__ import annotations

import math
import statistics

from src.agent.models import BuySignal, PairData, SellSignal

RULE_ID = "ou_spread_compression"

# Minimum hot-tier ticks required for a reliable OU fit
MIN_TICKS = 30

# Spread must exceed μ + THRESHOLD * σ_residual to be considered "wide"
THRESHOLD = 1.5

# Number of ticks over which the compression slope is computed
SLOPE_WINDOW = 5


MarketData = dict[str, PairData]


def _fit_ou(series: list[float]) -> tuple[float, float, float]:
    """OLS fit of an AR(1) to approximate OU parameters.

    Returns (mu, alpha, sigma_residual) where:
      alpha  = e^(-theta * dt)   — persistence coefficient
      mu     = long-run mean of the process
      sigma  = residual std dev (scale of the noise term)
    """
    x = series[:-1]
    y = series[1:]
    n = len(x)

    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)

    cov_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    var_x = sum((xi - x_mean) ** 2 for xi in x)

    if var_x == 0:
        return x_mean, 0.0, 0.0

    alpha = cov_xy / var_x
    # Clamp to (0, 1): alpha outside this range means the series is not mean-reverting
    alpha = max(1e-9, min(1 - 1e-9, alpha))
    beta = y_mean - alpha * x_mean
    mu = beta / (1 - alpha)

    residuals = [yi - (alpha * xi + beta) for xi, yi in zip(x, y)]
    sigma = statistics.stdev(residuals) if n >= 2 else 0.0

    return mu, alpha, sigma


def _slope(series: list[float]) -> float:
    """Least-squares slope of a short series."""
    n = len(series)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(series)
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, series))
    den = sum((xi - x_mean) ** 2 for xi in xs)
    return num / den if den else 0.0


def ou_spread_compression(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        ticks = pair_data.hot
        if len(ticks) < MIN_TICKS:
            continue

        spreads = [t.spread_rel for t in ticks]
        mu, alpha, sigma = _fit_ou(spreads)

        if sigma == 0 or not math.isfinite(mu):
            continue

        current_spread = spreads[-1]
        recent_slope = _slope(spreads[-SLOPE_WINDOW:])
        ts = ticks[-1].polled_at
        price = ticks[-1].last_price

        if current_spread > mu + THRESHOLD * sigma and recent_slope < 0:
            signals.append(BuySignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))
        elif current_spread < mu - THRESHOLD * sigma and recent_slope > 0:
            signals.append(SellSignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price))

    return signals
