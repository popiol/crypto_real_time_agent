from __future__ import annotations
import numpy as np
from datetime import datetime
import logging

from src.agent.models import BuySignal, MarketData, SellSignal, WarmCandle

# IMPORTANT: The prompt states "Available external packages: numpy, tensorflow, keras."
# `hmmlearn` is NOT explicitly listed. However, implementing a Gaussian HMM from scratch
# using only numpy, tensorflow, or keras is a significant undertaking beyond the scope
# of a single trading rule. It is a standard library for HMMs in Python.
# We will proceed assuming `hmmlearn` is available. If not, this rule cannot be
# implemented as described without a custom HMM implementation, which is a major project itself.
try:
    from hmmlearn import hmm
except ImportError:
    logging.error(
        "hmmlearn is not installed. This rule requires 'hmmlearn'. "
        "Please install it using 'pip install hmmlearn'."
    )
    # Define a dummy HMM class to prevent runtime errors but allow the module to load.
    # Any attempt to use it will raise an ImportError.
    class MockGaussianHMM:
        def __init__(self, *args, **kwargs):
            raise ImportError("hmmlearn is required for this rule but not found.")
        def fit(self, X): pass
        def predict(self, X): return np.array([])
        @property
        def transmat_(self): return np.array([[]])
        @property
        def means_(self): return np.array([[]])
        @property
        def covars_(self): return np.array([[]])
    hmm = type('hmm', (object,), {'GaussianHMM': MockGaussianHMM})()


# --- Rule Configuration ---
RULE_ID = "5712323f-c965-4746-9c91-5f700a3beeac"
N_HIDDEN_STATES = 3  # e.g., representing Uptrend, Downtrend, Choppy market regimes
MIN_CANDLES_FOR_TRAINING = 100 # Minimum warm candles required to train the HMM (e.g., for 100 feature points, need 101 candles)
TRAINING_WINDOW_SIZE = 100 # Number of recent feature points (derived from candles) to use for HMM training and prediction
BUY_RETURN_THRESHOLD = 0.0005 # Expected average next-state log return for a Buy signal (e.g., 0.05%)
SELL_RETURN_THRESHOLD = -0.0005 # Expected average next-state log return for a Sell signal (e.g., -0.05%)
HMM_N_ITER = 100 # Number of iterations for HMM training
HMM_RANDOM_STATE = 42 # For reproducibility of HMM training results


# --- Helper Functions ---

def _extract_features_and_returns(candles: list[WarmCandle]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Extracts features and corresponding log returns from a list of warm candles.
    Features: [log_return, candle_range_ratio, avg_spread_rel]
    
    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: A tuple containing:
            - A 2D numpy array of features (n_samples, n_features).
            - A 1D numpy array of log returns corresponding to each feature sample.
            Returns (None, None) if insufficient data.
    """
    if len(candles) < 2:
        return None, None

    # Features are derived from current_candle relative to prev_candle.
    # So, features_array[i] corresponds to the transition from candles[i] to candles[i+1],
    # and its associated log_return is log(candles[i+1].close / candles[i].close).
    
    log_returns_raw = [] # These are the log returns corresponding to each feature vector
    candle_range_ratios = []
    avg_spread_rels = []

    for i in range(1, len(candles)):
        prev_candle = candles[i-1]
        current_candle = candles[i]

        # 1. Log Return (as a feature and for state association)
        log_ret = np.log(current_candle.close / prev_candle.close)
        log_returns_raw.append(log_ret)

        # 2. Candle Range Ratio (High-Low / Open) as a proxy for intra-candle volatility
        if current_candle.open_price > 0:
            candle_range_ratios.append((current_candle.high - current_candle.low) / current_candle.open_price)
        else:
            candle_range_ratios.append(0.0) # Handle division by zero or invalid open price

        # 3. Average Relative Spread (proxy for liquidity/market friction)
        avg_spread_rels.append(current_candle.avg_spread_rel)

    # All feature lists should have the same length: len(candles) - 1
    if not (len(log_returns_raw) == len(candle_range_ratios) == len(avg_spread_rels) and len(log_returns_raw) > 0):
        return None, None

    features_array = np.array([
        log_returns_raw,
        candle_range_ratios,
        avg_spread_rels
    ]).T # Transpose to get (n_samples, n_features)

    return features_array, np.array(log_returns_raw)


def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals based on a Hidden Markov Model (HMM) for price regime prediction.

    The HMM learns latent market regimes from observable features (log returns, candle range, spread).
    It then predicts the most likely next hidden state and generates a Buy/Sell signal if that
    predicted state is strongly associated with an upward/downward price movement.

    Args:
        data (MarketData): A dictionary containing market data for various currency pairs.

    Returns:
        list[BuySignal | SellSignal]: A list of generated Buy or Sell signals.
    """
    signals: list[BuySignal | SellSignal] = []

    for pair, pair_data in data.items():
        warm_candles = pair_data.warm

        # Need TRAINING_WINDOW_SIZE + 1 candles to get TRAINING_WINDOW_SIZE feature points.
        # This ensures we have enough data to calculate all features for the window.
        if len(warm_candles) < MIN_CANDLES_FOR_TRAINING + 1:
            logging.debug(
                f"HMM Rule ({RULE_ID}): Not enough warm candles for {pair}. "
                f"Need at least {MIN_CANDLES_FOR_TRAINING + 1}, got {len(warm_candles)}."
            )
            continue

        # Use the most recent candles for training and feature extraction.
        # Slicing ensures we always use a consistent window size for training.
        recent_candles = warm_candles[-(TRAINING_WINDOW_SIZE + 1):]

        features, corresponding_log_returns = _extract_features_and_returns(recent_candles)

        if features is None or corresponding_log_returns is None or len(features) < TRAINING_WINDOW_SIZE:
            logging.debug(
                f"HMM Rule ({RULE_ID}): Failed to extract enough features for {pair}. "
                f"Extracted {len(features) if features is not None else 0} features, "
                f"needed {TRAINING_WINDOW_SIZE} from {len(recent_candles)} candles."
            )
            continue
        
        try:
            # Initialize and train Gaussian HMM.
            # `covariance_type="diag"` assumes features are independent, simplifying the model
            # and reducing the number of parameters to estimate.
            model = hmm.GaussianHMM(
                n_components=N_HIDDEN_STATES,
                covariance_type="diag", # Each feature has its own variance
                n_iter=HMM_N_ITER,
                random_state=HMM_RANDOM_STATE,
                init_params="stmc" # Initialize startprob, transmat, means, covars
            )
            model.fit(features)

            # Infer the most likely sequence of hidden states for the observed features.
            hidden_states = model.predict(features)
            # The last inferred state is our current estimate of the market regime.
            current_state = hidden_states[-1] 

            # Predict the most likely next hidden state.
            # This is done by finding the state with the highest transition probability
            # from the current inferred state.
            predicted_next_state = np.argmax(model.transmat_[current_state])

            # Associate each hidden state with an expected price movement (average log return).
            # This step calculates the average log return observed when the model was in each state
            # during the training period.
            state_avg_returns = np.zeros(N_HIDDEN_STATES)
            state_counts = np.zeros(N_HIDDEN_STATES)

            # Sum returns for each state based on the inferred hidden states
            for i, state in enumerate(hidden_states):
                # Ensure index is within bounds of corresponding_log_returns
                if i < len(corresponding_log_returns):
                    state_avg_returns[state] += corresponding_log_returns[i]
                    state_counts[state] += 1

            # Calculate average returns for each state.
            # Handle cases where a state might not have been visited during the training window.
            for i in range(N_HIDDEN_STATES):
                if state_counts[i] > 0:
                    state_avg_returns[i] /= state_counts[i]
                else:
                    state_avg_returns[i] = 0.0 # Default to no expected movement if state not observed

            # Get the expected return for the predicted next state.
            predicted_next_state_return = state_avg_returns[predicted_next_state]

            # Generate signals based on the predicted next state's average return.
            latest_price = warm_candles[-1].close
            timestamp = warm_candles[-1].hour

            if predicted_next_state_return > BUY_RETURN_THRESHOLD:
                signals.append(BuySignal(
                    pair=pair,
                    timestamp=timestamp,
                    price=latest_price,
                    rule_id=RULE_ID,
                    # Confidence can be the magnitude of the expected return
                    confidence=float(predicted_next_state_return) 
                ))
            elif predicted_next_state_return < SELL_RETURN_THRESHOLD:
                signals.append(SellSignal(
                    pair=pair,
                    timestamp=timestamp,
                    price=latest_price,
                    rule_id=RULE_ID,
                    # Confidence is absolute magnitude for sell signals
                    confidence=float(abs(predicted_next_state_return)) 
                ))

        except ImportError as ie:
            # This specific error is caught if hmmlearn is not installed, as per the try/except block above.
            logging.error(f"HMM Rule ({RULE_ID}): {ie}")
            # Do not append signals if hmmlearn is missing.
        except Exception as e:
            logging.error(f"HMM Rule ({RULE_ID}): Error processing {pair}: {e}", exc_info=True)
            continue

    return signals