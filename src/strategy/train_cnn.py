"""Train the 1D CNN model used by rule 10.

Usage:
    python -m src.strategy.train_cnn              # uses config.yaml
    python -m src.strategy.train_cnn path/to.yaml

Reads hourly Kraken Ticker snapshots from config.backtest_data_dir
(expected layout: year=YYYY/month=MM/day=DD/<unix_ts>.json).
Trains a small 1D CNN to predict whether the price will be ≥1% higher
in 24 hours, then saves the model to data/cnn_model/.

Feature vector per time step (2 features, matching rule_10_cnn_forecast.py):
  [close / mean_close_in_window,  (close_t - close_{t-1}) / close_{t-1}]

Label: 1 if price 24 h later > price_now * 1.01, else 0.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import yaml

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Hyperparameters ────────────────────────────────────────────────────────────
WINDOW_SIZE = 20
N_FEATURES = 2
FORECAST_HORIZON = 24   # hours ahead for the label
GAIN_THRESHOLD = 0.01   # 1% gain → positive label
BATCH_SIZE = 64
EPOCHS = 20
MIN_PAIR_COVERAGE = 0.8  # drop pairs present in < 80% of snapshots
MODEL_PATH = Path("data") / "rules" / "cnn" / "model.keras"


# ── Pair name normalisation (mirrors collector._normalise_pair) ────────────────

def _normalise(key: str) -> str:
    s = key
    if s.startswith("X") and len(s) >= 7:
        s = s[1:]
    if len(s) >= 6:
        for i in range(1, len(s) - 2):
            if s[i] == "Z" and s[i + 1 :].isupper() and len(s[i + 1 :]) == 3:
                s = s[:i] + s[i + 1 :]
                break
    return s


# ── Data loading ───────────────────────────────────────────────────────────────

def load_prices(backtest_dir: Path) -> dict[str, list[float]]:
    """Return {altname: [close_price, ...]} sorted chronologically."""
    files = sorted(
        f for f in backtest_dir.glob("**/*.json") if "_bidask" not in f.stem
    )
    if not files:
        raise FileNotFoundError(f"No snapshot files found in {backtest_dir}")
    logger.info("Found %d snapshot files", len(files))

    raw: dict[str, list[tuple[int, float]]] = {}   # altname → [(file_idx, price)]

    for idx, path in enumerate(files):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = payload.get("result", {})
        except Exception as exc:
            logger.debug("Skipping %s: %s", path, exc)
            continue

        for key, info in result.items():
            if key == "last":
                continue
            try:
                price = float(info["c"][0])
            except (KeyError, IndexError, ValueError, TypeError):
                continue

            altname = _normalise(key)
            if not altname.endswith("USD"):
                continue

            raw.setdefault(altname, []).append((idx, price))

    min_count = int(MIN_PAIR_COVERAGE * len(files))
    prices: dict[str, list[float]] = {}
    for pair, entries in raw.items():
        if len(entries) >= min_count:
            prices[pair] = [p for _, p in sorted(entries)]

    logger.info("Retained %d USD pairs with ≥%.0f%% coverage", len(prices), MIN_PAIR_COVERAGE * 100)
    return prices


# ── Feature / label construction ──────────────────────────────────────────────

def build_dataset(prices: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    X_list: list[list[list[float]]] = []
    y_list: list[int] = []

    for pair, series in prices.items():
        n = len(series)
        rocs = [0.0] + [
            (series[i] - series[i - 1]) / series[i - 1] if series[i - 1] else 0.0
            for i in range(1, n)
        ]

        for t in range(WINDOW_SIZE, n - FORECAST_HORIZON):
            window = series[t - WINDOW_SIZE : t]
            mean_p = sum(window) / WINDOW_SIZE
            if mean_p == 0:
                continue

            feats = [
                [p / mean_p, rocs[t - WINDOW_SIZE + i]]
                for i, p in enumerate(window)
            ]
            label = int(series[t + FORECAST_HORIZON] > series[t] * (1 + GAIN_THRESHOLD))

            X_list.append(feats)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ── Model definition ───────────────────────────────────────────────────────────

def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu",
                                   input_shape=(WINDOW_SIZE, N_FEATURES)),
            tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    backtest_dir = Path(cfg.get("backtest_data_dir", "../crypto_alerts_llm/data/raw"))
    if not backtest_dir.exists():
        logger.error("backtest_data_dir not found: %s", backtest_dir)
        sys.exit(1)

    logger.info("Loading prices from %s", backtest_dir)
    prices = load_prices(backtest_dir)

    if not prices:
        logger.error("No usable pair data found — aborting.")
        sys.exit(1)

    X, y = build_dataset(prices)
    logger.info(
        "Dataset: %d samples, positive rate %.1f%%", len(y), 100.0 * y.mean() if len(y) else 0
    )

    if len(X) < 200:
        logger.error("Too few samples (%d) to train reliably — aborting.", len(X))
        sys.exit(1)

    model = build_model()
    model.summary()

    model.fit(
        X,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        shuffle=True,
    )

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
