from __future__ import annotations

import statistics
from datetime import datetime
from math import log

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, WarmCandle


# --- Rule Configuration ---
N_PRICE_STATES = 3  # 0: Down, 1: Flat, 2: Up
N_VOL_STATES = 3    # 0: Low, 1: Medium, 2: High
N_COMPOSITE_STATES = N_PRICE_STATES * N_VOL_STATES

PRICE_CHANGE_THRESHOLD = 0.001  # 0.1% change to be considered 'Up' or 'Down'
VOL_PERIOD = 10                 # Number of warm candles for rolling volatility calculation (e.g., 10 hours)
SIGNAL_THRESHOLD = 0.6          # Probability must exceed this for a signal
MIN_WARM_CANDLES = VOL_PERIOD + 10 # Minimum warm candles needed: VOL_PERIOD for first vol calc + 10 for transitions
                                  # (e.g., 10 + 10 = 20 candles, approx 20 hours of data)

# --- Helper Functions ---

def _discretize_price_change(price_change_pct: float) -> int:
    """Assigns a price change percentage to a state: Down (0), Flat (1), Up (2)."""
    if price_change_pct < -PRICE_CHANGE_THRESHOLD:
        return 0  # Down
    elif price_change_pct > PRICE_CHANGE_THRESHOLD:
        return 2  # Up
    else:
        return 1  # Flat

def _calculate_volatility_series(candles: list[WarmCandle], period: int) -> list[float]:
    """Calculates rolling standard deviation of log returns.
    Returns a list of volatilities, where volatilities[i] corresponds to
    the volatility of candles[i + period].
    """
    if len(candles) < period + 1:
        return []

    log_returns = []
    for i in range(1, len(candles)):
        if candles[i-1].close > 0:
            log_returns.append(log(candles[i].close / candles[i-1].close))
        else:
            # If previous close is zero, log return is undefined. Treat as no change for volatility.
            log_returns.append(0.0) 

    volatilities = []
    # The first 'period' log_returns are needed for the first volatility point.
    # So, log_returns[0:period] gives the volatility for candles[period].
    for i in range(len(log_returns) - period + 1):
        window = log_returns[i : i + period]
        # np.std needs at least 2 elements for non-zero std. If all are same, std is 0.
        volatilities.append(np.std(window) if window else 0.0) 
    
    return volatilities

def _discretize_volatility(volatilities: list[float], n_states: int) -> list[int]:
    """Assigns each volatility value to a state using percentiles.
    Handles cases with insufficient unique volatility values.
    """
    if not volatilities:
        return []

    unique_vols = sorted(list(set(volatilities)))

    if len(unique_vols) == 1:
        # All volatilities are the same, assign all to the middle state
        return [n_states // 2] * len(volatilities)
    
    # Define bin edges using percentiles. We need n_states - 1 bin edges.
    # np.linspace(0, 100, n_states + 1)[1:-1] gives percentiles like [33.33, 66.66] for n_states=3
    percentile_points = np.linspace(0, 100, n_states + 1)[1:-1]
    
    # Ensure percentile points are within the range of unique values for robustness
    # This creates the bin edges.
    bins = np.percentile(unique_vols, percentile_points).tolist()

    states = []
    for vol in volatilities:
        state = 0
        # Assign state based on which bin the volatility falls into
        for b in bins:
            if vol > b:
                state += 1
        states.append(state)
    return states


def _transition_matrix(states: list[int], n_composite_states: int) -> list[list[float]]:
    """Build a row-normalised transition probability matrix from state sequence."""
    counts = [[0] * n_composite_states for _ in range(n_composite_states)]
    
    # Populate counts for transitions (state_A -> state_B)
    for a, b in zip(states, states[1:]):
        counts[a][b] += 1

    T: list[list[float]] = []
    # Normalize rows to get probabilities
    for row in counts:
        total = sum(row)
        T.append([c / total for c in row] if total > 0 else [0.0] * n_composite_states)
    return T


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on a Volatility-Aware Markov Chain.
    States are defined by a combination of price movement and volatility regime.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for volatility calculation and transitions
        if len(pair_data.warm) < MIN_WARM_CANDLES or not pair_data.hot:
            continue

        # --- 1. Calculate and discretize volatility series ---
        # The volatility series returned will have length `len(pair_data.warm) - VOL_PERIOD`.
        # Each element `volatilities[k]` corresponds to the volatility of `pair_data.warm[k + VOL_PERIOD]`.
        volatility_series = _calculate_volatility_series(pair_data.warm, VOL_PERIOD)
        if not volatility_series:
            continue # Not enough data for volatility calculation

        vol_states = _discretize_volatility(volatility_series, N_VOL_STATES)
        if not vol_states:
            continue # Should not happen if volatility_series is not empty

        # --- 2. Calculate and discretize price change series (aligned with volatility) ---
        # `warm_candles_for_states` are the candles for which we have both price change and volatility states.
        # This list starts from `pair_data.warm[VOL_PERIOD]`.
        warm_candles_for_states = pair_data.warm[VOL_PERIOD:]
        
        # This check ensures alignment, though it should hold true based on `_calculate_volatility_series` logic.
        if len(warm_candles_for_states) != len(vol_states):
            continue

        price_states = []
        for i in range(len(warm_candles_for_states)):
            current_candle = warm_candles_for_states[i]
            # The previous candle for the price change calculation is `pair_data.warm[VOL_PERIOD + i - 1]`.
            prev_candle = pair_data.warm[VOL_PERIOD + i - 1]
            
            if prev_candle.close > 0:
                price_change_pct = (current_candle.close - prev_candle.close) / prev_candle.close
            else:
                price_change_pct = 0.0 # Handle zero close price, treat as no change
            price_states.append(_discretize_price_change(price_change_pct))

        if len(price_states) != len(vol_states): # Final alignment check
            continue

        # --- 3. Combine price and volatility states into composite states ---
        # Each composite state is an integer from 0 to N_COMPOSITE_STATES - 1.
        # e.g., (price_state=0, vol_state=0) -> 0 * N_VOL_STATES + 0 = 0
        #       (price_state=2, vol_state=2) -> 2 * N_VOL_STATES + 2 = 8 (if N_VOL_STATES=3)
        composite_states = [
            ps * N_VOL_STATES + vs
            for ps, vs in zip(price_states, vol_states)
        ]

        # Need at least two composite states to observe a transition
        if len(composite_states) < 2:
            continue

        # --- 4. Construct Markov Chain Transition Matrix ---
        T = _transition_matrix(composite_states, N_COMPOSITE_STATES)

        # --- 5. Determine Current Composite State ---
        # The current state is the last calculated composite state from the warm candles.
        current_composite_state = composite_states[-1]

        # --- 6. Predict Probabilities for future price states ---
        # We need the timestamp and price from the latest hot tick for the signal.
        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price

        p_up_movement = 0.0
        p_down_movement = 0.0

        # Sum probabilities of transitioning from the current composite state
        # to any composite state that implies an 'Up' or 'Down' price movement.
        for next_composite_state_idx in range(N_COMPOSITE_STATES):
            prob = T[current_composite_state][next_composite_state_idx]
            
            # Extract the price state component from the next composite state.
            # Example: if N_VOL_STATES=3, state 7 (2*3+1) is price_state=2, vol_state=1
            predicted_price_state = next_composite_state_idx // N_VOL_STATES

            if predicted_price_state == 2: # Corresponds to 'Up' price state
                p_up_movement += prob
            elif predicted_price_state == 0: # Corresponds to 'Down' price state
                p_down_movement += prob

        # --- 7. Generate Signals ---
        if p_up_movement > SIGNAL_THRESHOLD:
            signals.append(BuySignal(pair=pair, timestamp=ts, price=price, confidence=p_up_movement))

        if p_down_movement > SIGNAL_THRESHOLD:
            signals.append(SellSignal(pair=pair, timestamp=ts, price=price, confidence=p_down_movement))

    return signals