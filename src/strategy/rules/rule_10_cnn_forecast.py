"""Rule 10 — Deep learning: 1D CNN price forecast (continuous online learning).

Each cycle this rule does three things in order:

  1. Label + train  — pending records that are ≥ FORECAST_HORIZON old are
                      labeled using the current live price; one gradient-descent
                      step is taken per labeled record via train_on_batch.
                      The model is saved to disk every SAVE_EVERY steps.

  2. Record         — if ≥ RECORD_INTERVAL has passed since the last record
                      for a pair, the current warm-tier feature window and live
                      price are appended to data/cnn_pending.ndjson.

  3. Infer          — run the model; emit BuySignal when P(gain) > SIGNAL_THRESHOLD,
                      SellSignal when P(gain) < 1 − SIGNAL_THRESHOLD.
                      Inference is suppressed until MIN_TRAINING_STEPS have
                      accumulated so random-weight outputs are never acted on.

No separate training script is needed. The model bootstraps itself from live
data and improves continuously. Use src/strategy/train_cnn.py to pre-train
from a historical dataset before going live.

State files:
    data/cnn_model/         TensorFlow SavedModel
    data/cnn_pending.ndjson pending training records (one JSON line each)
    data/cnn_state.json     {"training_steps": N}
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.agent.models import BuySignal, PairData, SellSignal, WarmCandle

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logger = logging.getLogger(__name__)

RULE_ID = "cnn_price_forecast"

WINDOW_SIZE = 20                        # warm candles per input window
N_FEATURES = 2                          # features per candle: (close_norm, roc)
FORECAST_HORIZON = timedelta(hours=24)
GAIN_THRESHOLD = 0.01                   # ≥1% gain in 24 h → label 1
SIGNAL_THRESHOLD = 0.6
MIN_TRAINING_STEPS = 10                 # suppress inference until this many steps done
RECORD_INTERVAL = timedelta(hours=1)    # record at most once per pair per hour
SAVE_EVERY = 10                         # persist model every N training steps

_MODEL_PATH = Path("data") / "cnn_model"
_PENDING_PATH = Path("data") / "cnn_pending.ndjson"
_STATE_PATH = Path("data") / "cnn_state.json"

MarketData = dict[str, PairData]

# ── Module-level state (persists for the lifetime of the process) ─────────────
_model: Any = None
_training_steps: int = 0
_last_recorded: dict[str, datetime] = {}


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_model():
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu",
                               input_shape=(WINDOW_SIZE, N_FEATURES)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def _load_model_and_state() -> None:
    global _model, _training_steps
    import tensorflow as tf

    if _MODEL_PATH.exists():
        try:
            _model = tf.keras.models.load_model(str(_MODEL_PATH))
            logger.info("CNN model loaded from %s", _MODEL_PATH)
        except Exception:
            logger.exception("Failed to load CNN model; starting fresh")
            _model = _build_model()
    else:
        logger.info("No CNN model found at %s; initializing with random weights", _MODEL_PATH)
        _model = _build_model()

    if _STATE_PATH.exists():
        try:
            _training_steps = json.loads(_STATE_PATH.read_text())["training_steps"]
        except Exception:
            _training_steps = 0


def _save_model_and_state() -> None:
    try:
        _model.save(str(_MODEL_PATH))
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps({"training_steps": _training_steps}))
    except Exception:
        logger.exception("Failed to save CNN model")


# ── Features ──────────────────────────────────────────────────────────────────

def _features(candles: list[WarmCandle]) -> list[list[float]]:
    """Return (close_norm, roc) per candle; first candle gets roc = 0."""
    closes = [c.close for c in candles]
    mean_close = statistics.mean(closes) or 1.0
    rows: list[list[float]] = []
    for i, close in enumerate(closes):
        prev = closes[i - 1]
        roc = (closes[i] - prev) / prev if i > 0 and prev != 0.0 else 0.0
        rows.append([close / mean_close, roc])
    return rows


# ── Pending records ───────────────────────────────────────────────────────────

def _load_pending() -> list[dict]:
    if not _PENDING_PATH.exists():
        return []
    records: list[dict] = []
    for line in _PENDING_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _save_pending(records: list[dict]) -> None:
    _PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in records)
    _PENDING_PATH.write_text(text + ("\n" if records else ""), encoding="utf-8")


# ── Label + train ─────────────────────────────────────────────────────────────

def _label_and_train(pending: list[dict], current_prices: dict[str, float]) -> list[dict]:
    """Label records old enough to have a known outcome; train one step each."""
    global _training_steps
    import numpy as np

    now = datetime.now(timezone.utc)
    remaining: list[dict] = []

    for rec in pending:
        recorded_at = datetime.fromisoformat(rec["recorded_at"])

        if now - recorded_at < FORECAST_HORIZON:
            remaining.append(rec)
            continue

        pair = rec["pair"]
        if pair not in current_prices:
            # Price unavailable this cycle; retry next cycle
            remaining.append(rec)
            continue

        label = 1.0 if current_prices[pair] > rec["price"] * (1 + GAIN_THRESHOLD) else 0.0

        X = np.array([rec["features"]], dtype=np.float32)
        y = np.array([label], dtype=np.float32)
        _model.train_on_batch(X, y)
        _training_steps += 1

        if _training_steps % SAVE_EVERY == 0:
            _save_model_and_state()

    return remaining


# ── Record ────────────────────────────────────────────────────────────────────

def _add_records(pending: list[dict], data: MarketData, now: datetime) -> list[dict]:
    """Append a pending record per pair, at most once per RECORD_INTERVAL."""
    for pair, pair_data in data.items():
        if len(pair_data.warm) < WINDOW_SIZE + 1 or not pair_data.hot:
            continue

        last = _last_recorded.get(pair)
        if last is not None and now - last < RECORD_INTERVAL:
            continue

        window = pair_data.warm[-(WINDOW_SIZE + 1):]
        feats = _features(window)[-WINDOW_SIZE:]

        pending.append({
            "pair": pair,
            "recorded_at": now.isoformat(),
            "price": pair_data.hot[-1].last_price,
            "features": feats,
        })
        _last_recorded[pair] = now

    return pending


# ── Infer ─────────────────────────────────────────────────────────────────────

def _infer(data: MarketData) -> list[BuySignal | SellSignal]:
    if _training_steps < MIN_TRAINING_STEPS:
        return []

    import tensorflow as tf

    signals: list[BuySignal | SellSignal] = []
    for pair, pair_data in data.items():
        if len(pair_data.warm) < WINDOW_SIZE + 1 or not pair_data.hot:
            continue

        window = pair_data.warm[-(WINDOW_SIZE + 1):]
        feats = _features(window)[-WINDOW_SIZE:]
        x = tf.constant([feats], dtype=tf.float32)
        prob = float(_model(x, training=False)[0, 0])
        ts = pair_data.hot[-1].polled_at
        price = pair_data.hot[-1].last_price

        if prob > SIGNAL_THRESHOLD:
            signals.append(BuySignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price, confidence=prob))
        elif prob < 1 - SIGNAL_THRESHOLD:
            signals.append(SellSignal(pair=pair, rule_id=RULE_ID, timestamp=ts, price=price, confidence=1 - prob))

    return signals


# ── Entry point ───────────────────────────────────────────────────────────────

def cnn_price_forecast(data: MarketData) -> list[BuySignal | SellSignal]:
    if _model is None:
        _load_model_and_state()

    now = datetime.now(timezone.utc)
    current_prices = {
        pair: pair_data.hot[-1].last_price
        for pair, pair_data in data.items()
        if pair_data.hot
    }

    pending = _load_pending()
    pending = _label_and_train(pending, current_prices)
    pending = _add_records(pending, data, now)
    _save_pending(pending)

    return _infer(data)
