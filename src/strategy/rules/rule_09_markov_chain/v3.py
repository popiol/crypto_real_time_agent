from __future__ import annotations

import statistics
from datetime import datetime

import numpy as np

from src.agent.models import BuySignal, MarketData, PairData, SellSignal, Tick, WarmCandle

# Parameters
N_ORDER = 3  # Order of the Markov chain (e.g., considering last 3 states)
WINDOW_SIZE_FOR_STATES = 50  # Window for calculating adaptive state bins (e.g., 50 bars)
NUM_STATES = 5  # Number of discrete price states
PROBABILITY_THRESHOLD = 0.6  # Probability threshold for buy/sell

# Minimum warm candles needed to build a meaningful transition matrix.
# We need enough warm candles such that `valid_historical_states` (after filtering Nones)
# has at least `N_ORDER + 1` elements to form at least one N_ORDER transition.
# `len(valid_historical_states) = len(all_warm_closes) - (WINDOW_SIZE_FOR_STATES - 1)`.
# So, `len(all_warm_closes) - (WINDOW_SIZE_FOR_STATES - 1) >= N_ORDER + 1`
# `len(all_warm_closes) >= N_ORDER + WINDOW_SIZE_FOR_STATES`.
MIN_WARM_CANDLES_FOR_TRAINING = N_ORDER + WINDOW_SIZE_FOR_STATES


def _get_adaptive_bins(prices_window: list[float], num_states: int) -> np.ndarray | None:
    """
    Calculates adaptive state bins based on a window of prices.
    The bins are dynamically set based on the mean and standard deviation of the window.
    """
    if len(prices_window) < 2:
        return None

    np_prices = np.array(prices_window)
    window_mean = np.mean(np_prices)
    window_std = np.std(np_prices)

    # Define the bounds for states, e.g., mean +/- 2 standard deviations (hence 4*std for total span)
    # Ensure a minimum range to avoid division by zero or single bin, even if std is tiny.
    # Using a small multiple of the mean ensures a relative range for prices.
    range_span = max(4 * window_std, 0.001 * abs(window_mean))
    if range_span == 0:  # Fallback for extremely flat data where mean is also zero (unlikely for prices)
        return None

    lower_bound = window_mean - range_span / 2
    upper_bound = window_mean + range_span / 2

    # np.linspace creates num_states + 1 boundaries for num_states bins
    return np.linspace(lower_bound, upper_bound, num_states + 1)


def _discretize_price(price: float, bins: np.ndarray) -> int | None:
    """
    Discretizes a single price into a state index [0, NUM_STATES-1].
    """
    if bins is None or len(bins) < 2:
        return None

    # np.digitize returns an index `i` such that `bins[i-1] <= price < bins[i]`.
    # It returns 0 for values less than bins[0] and len(bins) for values >= bins[len(bins)-1].
    # We want states from 0 to NUM_STATES-1.
    state = np.digitize(price, bins) - 1

    # Clamp the state to the valid range [0, NUM_STATES-1].
    # len(bins) - 2 corresponds to (NUM_STATES + 1) - 2 = NUM_STATES - 1.
    return max(0, min(state, len(bins) - 2))


def _build_transition_probabilities(
    valid_historical_states: list[int],
    n_order: int,
) -> dict[tuple[int, ...], dict[int, float]]:
    """
    Builds a higher-order transition probability matrix from a sequence of valid states.
    The keys for the outer dict are tuples representing the (N_ORDER-1) preceding states.
    The inner dict maps the next state to its probability.
    """
    transition_counts: dict[tuple[int, ...], dict[int, int]] = {}

    # Need at least N_ORDER states for the sequence + 1 for the next state
    if len(valid_historical_states) < n_order + 1:
        return {}  # Not enough data to form even one transition

    # Iterate through historical states to build sequences and next states.
    # A sequence is (state_t-N_ORDER, ..., state_t-1) and the next state is state_t.
    for i in range(n_order, len(valid_historical_states)):
        current_sequence = tuple(valid_historical_states[i - n_order : i])
        next_state = valid_historical_states[i]

        if current_sequence not in transition_counts:
            transition_counts[current_sequence] = {}
        if next_state not in transition_counts[current_sequence]:
            transition_counts[current_sequence][next_state] = 0
        transition_counts[current_sequence][next_state] += 1

    # Convert counts to probabilities
    transition_probabilities: dict[tuple[int, ...], dict[int, float]] = {}
    for seq, next_states_counts in transition_counts.items():
        total_transitions = sum(next_states_counts.values())
        if total_transitions > 0:
            transition_probabilities[seq] = {
                state: count / total_transitions for state, count in next_states_counts.items()
            }
        # If total_transitions is 0, the sequence has no observed transitions,
        # so it's left out or has an empty probability dict (handled by initial check).

    return transition_probabilities


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        # Ensure we have enough warm candles for training and current context
        if len(pair_data.warm) < MIN_WARM_CANDLES_FOR_TRAINING or not pair_data.hot:
            continue

        all_warm_closes = [c.close for c in pair_data.warm]
        current_hot_tick = pair_data.hot[-1]
        current_hot_price = current_hot_tick.last_price
        current_timestamp = current_hot_tick.polled_at

        # --- 1. Generate historical states for training the model ---
        # This list will contain states for each warm candle.
        # Initial candles (where WINDOW_SIZE_FOR_STATES prices are not yet available) will have None.
        historical_states_for_training: list[int | None] = []
        for i in range(len(all_warm_closes)):
            if i >= WINDOW_SIZE_FOR_STATES - 1:
                # Window ends at current warm candle 'i'
                current_window_prices = all_warm_closes[i - WINDOW_SIZE_FOR_STATES + 1 : i + 1]
                bins = _get_adaptive_bins(current_window_prices, NUM_STATES)
                if bins is not None:
                    state = _discretize_price(all_warm_closes[i], bins)
                    historical_states_for_training.append(state)
                else:
                    historical_states_for_training.append(None)
            else:
                historical_states_for_training.append(None)

        # Filter out None states to get a clean sequence for building the transition matrix
        valid_historical_states = [s for s in historical_states_for_training if s is not None]

        # Ensure enough valid states to build transition probabilities
        if len(valid_historical_states) < N_ORDER + 1:
            continue # Not enough data to form even one N_ORDER transition

        # --- 2. Build higher-order transition probabilities ---
        transition_probabilities = _build_transition_probabilities(
            valid_historical_states, N_ORDER
        )

        if not transition_probabilities:  # If no transitions could be learned
            continue

        # --- 3. Generate Signal for the current hot price ---
        # Get prices for the current adaptive bin calculation:
        # last (WINDOW_SIZE_FOR_STATES - 1) warm closes + current hot price.
        # Ensure enough warm closes for this window.
        if len(all_warm_closes) < WINDOW_SIZE_FOR_STATES - 1:
            continue
        prices_for_current_bins = all_warm_closes[-(WINDOW_SIZE_FOR_STATES - 1):] + [current_hot_price]

        current_bins = _get_adaptive_bins(prices_for_current_bins, NUM_STATES)
        if current_bins is None:
            continue

        current_state = _discretize_price(current_hot_price, current_bins)
        if current_state is None:
            continue

        # Form the current state sequence for prediction.
        # This sequence should be the last N_ORDER-1 historical states (from warm data)
        # followed by the current state (from hot data).
        num_prev_states_needed = N_ORDER - 1

        if len(valid_historical_states) < num_prev_states_needed:
            continue  # Not enough historical states to form the sequence for prediction

        # Extract the last `num_prev_states_needed` elements from `valid_historical_states`.
        # The slice `max(0, len(list) - num_elements):` correctly handles cases where
        # `num_elements` is 0 (for N_ORDER=1) or larger than the list length.
        historical_states_for_prediction_sequence = valid_historical_states[
            max(0, len(valid_historical_states) - num_prev_states_needed):
        ]

        # The full sequence for lookup in the transition matrix
        current_sequence_for_prediction = tuple(historical_states_for_prediction_sequence + [current_state])

        if current_sequence_for_prediction in transition_probabilities:
            probs = transition_probabilities[current_sequence_for_prediction]

            # Calculate probability of moving to a higher or lower state
            prob_up = sum(p for state, p in probs.items() if state > current_state)
            prob_down = sum(p for state, p in probs.items() if state < current_state)

            if prob_up > PROBABILITY_THRESHOLD:
                signals.append(BuySignal(pair=pair, timestamp=current_timestamp, price=current_hot_price))
            elif prob_down > PROBABILITY_THRESHOLD:
                signals.append(SellSignal(pair=pair, timestamp=current_timestamp, price=current_hot_price))

    return signals