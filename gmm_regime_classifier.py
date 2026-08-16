import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 1: FETCH DATA & FEATURE ENGINEERING
ticker = "SPY"
data = yf.download(ticker, start="2015-01-01", end="2024-01-01", progress=False)

# Flatten columns if yfinance returns a MultiIndex (handles single or multiple levels)
if isinstance(data.columns, pd.MultiIndex):
    # If ticker is one of the levels, extract it; otherwise drop the ticker level
    if ticker in data.columns.levels[1]:
        df = data.xs(ticker, level=1, axis=1).copy()
    elif ticker in data.columns.levels[0]:
        df = data.xs(ticker, level=0, axis=1).copy()
    else:
        df = data.copy()
        df.columns = df.columns.get_level_values(0)
else:
    df = data.copy()

# Fallback: if 'Adj Close' doesn't exist, use 'Close'
price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

# Feature 1: Log Returns
df["log_ret"] = np.log(df[price_col] / df[price_col].shift(1))

# Feature 2: 20-Day Annualized Realized Volatility
df["vol_20d"] = df["log_ret"].rolling(window=20).std() * np.sqrt(252)

# Feature 3: Normalized Average True Range (14-day ATR / Close)
high_low = df["High"] - df["Low"]
high_close_prev = (df["High"] - df["Close"].shift(1)).abs()
low_close_prev = (df["Low"] - df["Close"].shift(1)).abs()
true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
df["norm_atr"] = (true_range.rolling(window=14).mean()) / df["Close"]

df.dropna(inplace=True)

# Expanding-Window Standardization
features = ["log_ret", "vol_20d", "norm_atr"]
X = pd.DataFrame(index=df.index)

for col in features:
    exp_mean = df[col].expanding(min_periods=60).mean()
    exp_std = df[col].expanding(min_periods=60).std()
    X[col] = (df[col] - exp_mean) / exp_std

valid_mask = X.notna().all(axis=1)
df = df.loc[valid_mask].copy()
X_scaled = X.loc[valid_mask].values

# 2: MODEL TUNING VIA BIC MINIMIZATION
k_candidates = range(2, 7)
bic_scores = []
models = {}

for k in k_candidates:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(X_scaled)
    bic = gmm.bic(X_scaled)
    bic_scores.append(bic)
    models[k] = gmm

optimal_k = k_candidates[np.argmin(bic_scores)]
print(f"Optimal clusters by BIC: K = {optimal_k}")

# 3: FIT OPTIMAL GMM & DETERMINISTIC SORTING
best_gmm = models[optimal_k]
raw_probs = best_gmm.predict_proba(X_scaled)

# Sort clusters deterministically by empirical mean returns
cluster_mean_returns = best_gmm.means_[:, 0]
sort_order = np.argsort(cluster_mean_returns)
sorted_probs = raw_probs[:, sort_order]

# 4: EMA PROBABILITY SMOOTHING
prob_cols = [f"P_state_{i}" for i in range(optimal_k)]
prob_df = pd.DataFrame(sorted_probs, index=df.index, columns=prob_cols)

# Apply Exponential Moving Average (span=4)
smoothed_prob_df = prob_df.ewm(span=4, adjust=False).mean()

# Re-normalize across states so probabilities sum to 1.0 each day
smoothed_prob_df = smoothed_prob_df.div(smoothed_prob_df.sum(axis=1), axis=0)

for col in prob_cols:
    df[col] = smoothed_prob_df[col]

# 5: DYNAMIC WEIGHT ALLOCATION & EVALUATION
weights_per_state = np.linspace(0.0, 1.0, optimal_k)
df["target_weight"] = np.dot(smoothed_prob_df.values, weights_per_state)

# Shift weight by 1 bar to ensure no lookahead bias
df["strat_ret"] = df["target_weight"].shift(1) * df["log_ret"]
df.dropna(inplace=True)

# Cumulative Returns
df["cum_bh"] = np.exp(df["log_ret"].cumsum())
df["cum_strat"] = np.exp(df["strat_ret"].cumsum())

# Sharpe Ratios
sharpe_bh = (df["log_ret"].mean() / df["log_ret"].std()) * np.sqrt(252)
sharpe_strat = (df["strat_ret"].mean() / df["strat_ret"].std()) * np.sqrt(252)

# Maximum Drawdown
def calculate_mdd(cum_series):
    peak = cum_series.cummax()
    drawdown = (cum_series - peak) / peak
    return drawdown.min()

mdd_bh = calculate_mdd(df["cum_bh"])
mdd_strat = calculate_mdd(df["cum_strat"])

print("\nPERFORMANCE SUMMARY")
print(f"\nBuy & Hold Sharpe Ratio:       {sharpe_bh:.2f}")
print(f"GMM Strategy Sharpe Ratio:     {sharpe_strat:.2f}")
print(f"Buy & Hold Max Drawdown:       {mdd_bh * 100:.2f}%")
print(f"GMM Strategy Max Drawdown:     {mdd_strat * 100:.2f}%")

# VISUALIZATION
# Compute dominant regime directly from the aligned df columns
prob_matrix = df[prob_cols].values
dominant_regime = prob_matrix.argmax(axis=1)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# 1. Price with Dominant Regime Overlay
axes[0].plot(df.index, df[price_col], color="black", alpha=0.6, label="SPY Price")
axes[0].scatter(df.index, df[price_col], c=dominant_regime, cmap="coolwarm", s=6, alpha=0.8)
axes[0].set_title(f"SPY Price & Dominant Regime Overlay (K={optimal_k})", fontsize=12)
axes[0].set_ylabel("Price ($)")
axes[0].grid(True, alpha=0.3)

# 2. Dynamic Portfolio Allocation Weight
axes[1].plot(df.index, df["target_weight"], color="teal", lw=1.2, label="Dynamic Weight")
axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
axes[1].axhline(0.0, color="gray", linestyle="--", alpha=0.5)
axes[1].set_title("Smoothed Dynamic Portfolio Weight Allocation", fontsize=12)
axes[1].set_ylabel("Weight (0 to 1)")
axes[1].grid(True, alpha=0.3)

# 3. Cumulative Equity Comparison
axes[2].plot(df.index, df["cum_strat"], color="darkblue", lw=1.5, label=f"GMM Strategy (Sharpe: {sharpe_strat:.2f}, MDD: {mdd_strat*100:.1f}%)")
axes[2].plot(df.index, df["cum_bh"], color="gray", linestyle="--", lw=1.2, label=f"Buy & Hold SPY (Sharpe: {sharpe_bh:.2f}, MDD: {mdd_bh*100:.1f}%)")
axes[2].set_title("Cumulative Growth of $1 Investment", fontsize=12)
axes[2].set_ylabel("Growth ($)")
axes[2].set_xlabel("Date")
axes[2].legend(loc="upper left")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()