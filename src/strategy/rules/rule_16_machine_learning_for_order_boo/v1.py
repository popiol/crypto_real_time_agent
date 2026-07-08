from __future__ import annotations
import collections
from datetime import datetime
import numpy as np
import statistics

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Data models
from src.agent.models import BuySignal, MarketData, SellSignal, Tick, OrderBook


# Define a dummy OrderBookLevel class for type hinting, assuming its structure
# This assumes OrderBookLevel objects have 'price' and 'volume' attributes.
class OrderBookLevel:
    price: float
    volume: float

    def __init__(self, price: float, volume: float):
        self.price = price
        self.volume = volume


# Global variables for the model and scaler
# These will be initialized once and reused across signal calls.
_model: keras.Model | None = None
_scaler: StandardScaler | None = None
_model_initialized: bool = False

# --- Constants for the rule ---
RULE_ID = "e8db0015-2944-4b94-81c3-c95eb948e192"
MIN_TICKS_FOR_FEATURES = 60  # Minimum ticks required to engineer features (e.g., 1 minute of 1-second ticks)
ORDER_BOOK_DEPTH_TO_USE = 5  # Using up to 5 levels from the OrderBook as described
BUY_THRESHOLD = 0.70  # Probability threshold for a buy signal
SELL_THRESHOLD = 0.70  # Probability threshold for a sell signal
MIN_PROFIT_TARGET_MAGNITUDE = 0.0005  # Minimum predicted price movement (e.g., 0.05% of price)
LOOKBACK_TICKS_SHORT = 10  # For short-term changes (e.g., 10 seconds if ticks are 1s)
LOOKBACK_TICKS_MEDIUM = 30  # For medium-term changes (e.g., 30 seconds if ticks are 1s)

# --- Helper Functions ---

def _get_order_book_levels(ob: OrderBook | None, is_bid: bool, depth: int) -> list[OrderBookLevel]:
    """Safely retrieves a specified depth of order book levels."""
    if ob is None:
        return []
    levels = ob.bids if is_bid else ob.asks
    # Ensure levels are sorted by price (bids descending, asks ascending)
    # The given OrderBook structure doesn't specify sorting, assuming it's already sorted
    # or that it provides best (level 1) first.
    return levels[:depth]

def _train_model_placeholder():
    """
    A placeholder function to simulate model training.
    In a real system, this would:
    1. Load extensive historical data.
    2. Engineer features for each historical snapshot.
    3. Define a target variable (e.g., future price change over X minutes, binarized).
    4. Train the ML model (e.g., LightGBM, XGBoost, or a neural network).
    5. Save the trained model and scaler to disk.
    6. Implement regular retraining logic.

    For this exercise, it creates a simple Keras model and a dummy StandardScaler
    to allow the `signal` function to run without actual training data.
    """
    global _model, _scaler, _model_initialized

    # The number of features must match what _extract_features generates.
    # Count: 4 (basic) + 4 (short changes) + 2 (medium changes) + 3 (WAP) +
    #        9 (OBI & Cum Vol for 3 depths) + 1 (VWAP dev) + 6 (Order Flow Velocity for 3 depths) = 29 features
    N_FEATURES = 29
    N_SAMPLES = 1000  # Dummy samples for training

    # Create dummy features for scaler fitting
    X_train = np.random.rand(N_SAMPLES, N_FEATURES)

    # Initialize and fit scaler
    _scaler = StandardScaler()
    _scaler.fit(X_train)

    # Build a simple Keras model (Feed-forward Neural Network)
    model = keras.Sequential([
        keras.layers.Input(shape=(N_FEATURES,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        # Two outputs for P(Up) and P(Down) using sigmoid activation for probabilities
        keras.layers.Dense(2, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Create dummy target for training (e.g., [P(Up), P(Down)])
    y_train_up = np.random.randint(0, 2, N_SAMPLES)
    y_train_down = np.random.randint(0, 2, N_SAMPLES)
    y_train = np.column_stack([y_train_up, y_train_down])

    # Fit the model (silently for this placeholder)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    _model = model
    _model_initialized = True
    # print("ML model and scaler initialized (placeholder training).")


def _extract_features(ticks: collections.deque[Tick]) -> np.ndarray | None:
    """
    Extracts a rich set of machine learning features from a deque of historical ticks.
    Returns a numpy array of features for the latest tick, or None if insufficient data.
    """
    if len(ticks) < MIN_TICKS_FOR_FEATURES:
        return None

    latest_tick = ticks[-1]
    features = []

    # 1. Basic Metrics (from latest tick)
    mid_price = latest_tick.mid_price
    spread_abs = latest_tick.spread_abs
    spread_rel = latest_tick.spread_rel
    bid_vol = latest_tick.bid_volume
    ask_vol = latest_tick.ask_volume

    features.extend([mid_price, spread_abs, spread_rel])
    features.append(bid_vol / ask_vol if ask_vol > 0 else 0.0)  # Bid/Ask Volume Ratio

    # 2. Short-term Changes (requires historical ticks)
    # Ensure enough ticks are available for lookback
    if len(ticks) >= LOOKBACK_TICKS_SHORT + 1:
        tick_short_ago = ticks[-LOOKBACK_TICKS_SHORT - 1]
        features.append(mid_price - tick_short_ago.mid_price)
        features.append(spread_abs - tick_short_ago.spread_abs)
        features.append(bid_vol - tick_short_ago.bid_volume)
        features.append(ask_vol - tick_short_ago.ask_volume)
    else:
        features.extend([0.0] * 4)  # Fill with zeros if not enough history

    if len(ticks) >= LOOKBACK_TICKS_MEDIUM + 1:
        tick_medium_ago = ticks[-LOOKBACK_TICKS_MEDIUM - 1]
        features.append(mid_price - tick_medium_ago.mid_price)
        features.append(spread_abs - tick_medium_ago.spread_abs)
    else:
        features.extend([0.0] * 2)  # Fill with zeros

    # 3. Order Book Features (from latest tick's order_book)
    bid_levels = _get_order_book_levels(latest_tick.order_book, True, ORDER_BOOK_DEPTH_TO_USE)
    ask_levels = _get_order_book_levels(latest_tick.order_book, False, ORDER_BOOK_DEPTH_TO_USE)

    # Pad levels if not enough depth to ensure consistent feature vector size
    while len(bid_levels) < ORDER_BOOK_DEPTH_TO_USE:
        bid_levels.append(OrderBookLevel(price=0.0, volume=0.0))
    while len(ask_levels) < ORDER_BOOK_DEPTH_TO_USE:
        ask_levels.append(OrderBookLevel(price=0.0, volume=0.0))

    # Weighted Average Prices (WAP)
    sum_bid_pv = sum(level.price * level.volume for level in bid_levels if level.volume > 0)
    sum_bid_v = sum(level.volume for level in bid_levels)
    sum_ask_pv = sum(level.price * level.volume for level in ask_levels if level.volume > 0)
    sum_ask_v = sum(level.volume for level in ask_levels)

    wap_bid = sum_bid_pv / sum_bid_v if sum_bid_v > 0 else 0.0
    wap_ask = sum_ask_pv / sum_ask_v if sum_ask_v > 0 else 0.0
    # Fallback to mid_price if no depth for WAP calculation
    wap_mid = (wap_bid + wap_ask) / 2 if wap_bid > 0 and wap_ask > 0 else mid_price

    features.extend([wap_bid, wap_ask, wap_mid])

    # Order Book Imbalance (OBI) & Cumulative Volumes
    for k in [1, 3, ORDER_BOOK_DEPTH_TO_USE]:
        current_bid_vols = [level.volume for level in bid_levels[:k]]
        current_ask_vols = [level.volume for level in ask_levels[:k]]
        cum_bid_vol_k = sum(current_bid_vols)
        cum_ask_vol_k = sum(current_ask_vols)
        total_vol_k = cum_bid_vol_k + cum_ask_vol_k
        obi_k = (cum_bid_vol_k - cum_ask_vol_k) / total_vol_k if total_vol_k > 0 else 0.0
        features.extend([obi_k, cum_bid_vol_k, cum_ask_vol_k])

    # Volume-Weighted Average Price (VWAP) deviations from mid-price (using order book depth)
    total_ob_volume = sum_bid_v + sum_ask_v
    if total_ob_volume > 0:
        ob_vwap_mid = (sum_bid_pv + sum_ask_pv) / total_ob_volume
        features.append(ob_vwap_mid - mid_price)
    else:
        features.append(0.0)

    # 4. Order Flow Velocity (Requires historical order book data)
    # Compare cumulative volumes from current tick to LOOKBACK_TICKS_SHORT ago
    if len(ticks) >= LOOKBACK_TICKS_SHORT + 1:
        prev_tick = ticks[-LOOKBACK_TICKS_SHORT - 1]
        prev_bid_levels = _get_order_book_levels(prev_tick.order_book, True, ORDER_BOOK_DEPTH_TO_USE)
        prev_ask_levels = _get_order_book_levels(prev_tick.order_book, False, ORDER_BOOK_DEPTH_TO_USE)

        # Pad previous levels for consistent calculation
        while len(prev_bid_levels) < ORDER_BOOK_DEPTH_TO_USE:
            prev_bid_levels.append(OrderBookLevel(price=0.0, volume=0.0))
        while len(prev_ask_levels) < ORDER_BOOK_DEPTH_TO_USE:
            prev_ask_levels.append(OrderBookLevel(price=0.0, volume=0.0))

        for k in [1, 3, ORDER_BOOK_DEPTH_TO_USE]:
            prev_cum_bid_vol_k = sum(level.volume for level in prev_bid_levels[:k])
            prev_cum_ask_vol_k = sum(level.volume for level in prev_ask_levels[:k])

            current_cum_bid_vol_k = sum(level.volume for level in bid_levels[:k])
            current_cum_ask_vol_k = sum(level.volume for level in ask_levels[:k])

            delta_cum_bid_vol_k = current_cum_bid_vol_k - prev_cum_bid_vol_k
            delta_cum_ask_vol_k = current_cum_ask_vol_k - prev_cum_ask_vol_k
            features.extend([delta_cum_bid_vol_k, delta_cum_ask_vol_k])
    else:
        features.extend([0.0] * 6)  # 3 depths * 2 (bid/ask)

    return np.array(features, dtype=np.float32)


# --- Main Signal Function ---

def signal(data: MarketData) -> list[BuySignal | SellSignal]:
    """
    Generates trading signals for currency pairs based on a machine learning model
    analyzing real-time order book dynamics.

    The model is assumed to be pre-trained and loaded/initialized once.
    It predicts probabilities of short-term upward or downward price movements.
    """
    global _model, _scaler, _model_initialized

    signals: list[BuySignal | SellSignal] = []

    # Initialize model and scaler if not already done
    if not _model_initialized:
        _train_model_placeholder()  # This builds and 'trains' a dummy model and scaler

    if _model is None or _scaler is None:
        # This should ideally not happen if _train_model_placeholder is successful.
        # Return empty signals if model or scaler are not ready.
        return signals

    for pair, pair_data in data.items():
        # Use collections.deque for efficient historical data access
        ticks = collections.deque(pair_data.hot)

        # Ensure enough historical ticks are available for feature engineering
        if len(ticks) < MIN_TICKS_FOR_FEATURES:
            continue

        # Extract features for the latest tick
        features = _extract_features(ticks)

        if features is None:  # Not enough data for feature extraction
            continue

        # Reshape features for the scaler and model (1 sample, N features)
        features_reshaped = features.reshape(1, -1)

        # Scale features using the pre-fitted scaler
        scaled_features = _scaler.transform(features_reshaped)

        # Predict probabilities using the trained model
        # The model outputs two probabilities: P(Up), P(Down)
        predictions = _model.predict(scaled_features, verbose=0)
        up_prob = predictions[0][0]  # Probability of upward movement
        down_prob = predictions[0][1] # Probability of downward movement

        # Simulate predicted magnitude from probabilities (simplified)
        # In a real system, the model might predict actual magnitude or
        # have different output classes (e.g., 'small_up', 'large_up').
        # Here, we use confidence scaled by a conceptual target magnitude.
        predicted_up_magnitude = up_prob * MIN_PROFIT_TARGET_MAGNITUDE * 2
        predicted_down_magnitude = down_prob * MIN_PROFIT_TARGET_MAGNITUDE * 2

        latest_tick = ticks[-1]
        current_price = latest_tick.last_price
        polled_at = latest_tick.polled_at

        # Signal Generation Logic
        if up_prob > BUY_THRESHOLD and predicted_up_magnitude > MIN_PROFIT_TARGET_MAGNITUDE:
            signals.append(BuySignal(
                pair=pair,
                timestamp=polled_at,
                price=current_price,
                rule_id=RULE_ID,
                confidence=up_prob
            ))
        elif down_prob > SELL_THRESHOLD and predicted_down_magnitude > MIN_PROFIT_TARGET_MAGNITUDE:
            signals.append(SellSignal(
                pair=pair,
                timestamp=polled_at,
                price=current_price,
                rule_id=RULE_ID,
                confidence=down_prob
            ))

    return signals