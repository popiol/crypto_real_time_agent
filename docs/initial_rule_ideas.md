# Initial Rule Ideas

Seed ideas for the first version of `strategy.py`. Each idea is described at the concept level; implementation details will be worked out when an idea is selected and implemented by the Strategy Updater pipeline.

---

## 1. Hardcoded heuristic — spread compression spike

**Approach**: purely deterministic, no model.  
**Idea**: compute a rolling baseline of the bid/ask spread over the last N hot-tier ticks. When the current spread drops more than X% below the baseline (i.e. the market is unusually tight), emit a buy signal. A tight spread can indicate strong buying pressure compressing the ask side.  
**Data needed**: hot tier (recent ticks), spread history.  
**Pros**: zero latency, fully explainable, no training required.  
**Cons**: fragile to parameter choice; spread alone is a weak signal.

---

## 2. Mean reversion — Bollinger Band lower touch

**Approach**: classical statistical signal.  
**Idea**: compute a rolling mean and standard deviation of the last-trade price over the warm tier (hourly OHLC). Emit a buy signal when the current price crosses below the lower band (mean − k·σ), indicating the price has deviated unusually far downward and may revert.  
**Data needed**: warm tier.  
**Pros**: well-understood, widely used baseline.  
**Cons**: fails in trending markets; requires careful band width tuning.

---

## 3. Stochastic process — mean-reverting spread (Ornstein–Uhlenbeck)

**Approach**: fit an Ornstein–Uhlenbeck (OU) process to the bid/ask spread time series.  
**Idea**: the OU model gives a long-run mean and a speed-of-reversion parameter. When the current spread is significantly above its long-run mean (i.e. the market is unusually wide), the OU model predicts it will compress — which often precedes a price move. Emit a buy signal when the spread is above the OU mean by more than a threshold and the spread derivative is already turning negative (compression beginning).  
**Data needed**: hot tier and warm tier for fitting.  
**Pros**: principled probabilistic model of spread dynamics.  
**Cons**: requires periodic refitting; spread alone is an indirect signal.

---

## 4. Time series — ARIMA price forecast

**Approach**: fit an ARIMA(p,d,q) model to the hourly closing prices in the warm tier.  
**Idea**: forecast the next 1–3 hourly prices. Emit a buy signal when the forecast shows a statistically significant upward move relative to the current price.  
**Data needed**: warm tier (hourly close prices).  
**Pros**: captures autocorrelation and trend; interpretable.  
**Cons**: ARIMA assumes linearity and stationarity; crypto prices are highly non-stationary. Requires ADF test and differencing. Re-fitting every cycle is expensive.

---

## 5. Signal processing — FFT dominant cycle detection

**Approach**: apply a Fast Fourier Transform to the warm-tier hourly price series.  
**Idea**: identify the dominant frequency components in the price series. If the dominant cycle suggests the price is currently near a trough (phase analysis), emit a buy signal. Can also use the FFT to filter out noise before applying other rules.  
**Data needed**: warm tier.  
**Pros**: captures periodic patterns (e.g. daily cycles) that other methods miss.  
**Cons**: crypto cycles are unstable and non-stationary; FFT assumes stationarity. Works best as a pre-filter rather than a standalone signal.

---

## 6. Signal processing — Kalman filter price tracker

**Approach**: use a Kalman filter to maintain an optimal real-time estimate of the true price and its velocity (rate of change), filtering out tick noise.  
**Idea**: the Kalman filter produces a smoothed price estimate and a velocity estimate. Emit a buy signal when the filtered velocity crosses from negative to positive (estimated price momentum reversal) while the price is below its recent warm-tier average.  
**Data needed**: hot tier (real-time ticks for filter update), warm tier (for baseline).  
**Pros**: optimal noise filtering; real-time, low latency; no batch refitting needed.  
**Cons**: requires tuning of process and measurement noise covariance matrices.

---

## 7. Market microstructure — order book imbalance

**Approach**: use bid/ask depth data from the hot tier.  
**Idea**: compute the order book imbalance ratio: `(bid_volume − ask_volume) / (bid_volume + ask_volume)` across the top N levels. A strongly positive imbalance (more buy-side depth) predicts short-term upward price pressure. Emit a buy signal when the imbalance exceeds a threshold and has been sustained for at least M consecutive ticks.  
**Data needed**: hot tier (bid/ask volumes per tick).  
**Pros**: forward-looking (reflects pending orders, not just past trades); low latency.  
**Cons**: order book can be spoofed; only top-of-book is available from the Ticker endpoint (full depth would require WebSocket order book feed).

---

## 8. Momentum — rate-of-change threshold

**Approach**: simple trend-following.  
**Idea**: compute the rate of change (ROC) of the price over the last N hot-tier ticks and over the last M warm-tier hours. Emit a buy signal when the short-term ROC is positive and accelerating, and the medium-term ROC has recently turned from negative to positive (momentum regime change).  
**Data needed**: hot tier, warm tier.  
**Pros**: captures breakout entries; simple and fast.  
**Cons**: prone to false signals in choppy markets; needs a volatility filter to avoid noise.

---

## 9. Markov chain — price-level transition probability

**Approach**: model price movements as a discrete Markov chain.  
**Idea**: discretise the warm-tier price history into states (e.g. percentage bins relative to a rolling mean). Build a transition probability matrix from historical state sequences. Given the current state, look up the probability of moving to a higher state in the next step. Emit a buy signal when this probability exceeds a threshold.  
**Data needed**: warm and cold tiers (more data = better transition estimates).  
**Pros**: non-parametric; captures asymmetric up/down transition probabilities.  
**Cons**: sensitive to state discretisation; stationarity assumption is weak for crypto.

---

## 10. Deep learning — 1D convolutional neural network (CNN)

**Approach**: train a small 1D CNN on sequences of hot/warm tier features to classify whether the price will be higher in N hours.  
**Idea**: construct a fixed-length feature vector per time step (price, spread, volume, ROC, imbalance). Train a 1D CNN on labelled windows from the backtesting dataset where the label is `1` if `gain_Nh > threshold` else `0`. The trained model outputs a probability; emit a buy signal when probability exceeds a threshold.  
**Data needed**: backtesting dataset for training; hot and warm tiers for inference.  
**Pros**: can learn non-linear patterns across multiple time scales simultaneously.  
**Cons**: requires labelled training data and a training pipeline; model needs periodic retraining; black-box.

---

## 11. Reinforcement learning — Q-learning agent

**Approach**: train a small RL agent on the backtesting dataset.  
**Idea**: define a state space from recent market features and a reward signal based on the 24h gain after a buy action. Train a Q-learning (or DQN) agent to maximise cumulative reward. At inference time, emit a buy signal when the agent's policy selects the "buy" action.  
**Data needed**: backtesting dataset for training; live tiers for inference.  
**Pros**: directly optimises for profit rather than a proxy metric; adapts to market dynamics through retraining.  
**Cons**: most complex to implement and retrain; reward signal sparsity (only one outcome per 24h window) makes training slow.
