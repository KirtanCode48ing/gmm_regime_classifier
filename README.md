# Market Regime Classifier

An unsupervised quantitative machine learning pipeline that identifies latent market regimes (Bull, Bear, Chop) on **SPY (2015–2024)** using **Gaussian Mixture Models (GMM)** and dynamically scales exposure via **EMA-smoothed posterior probabilities**.

---

## Key Results vs. Buy & Hold

| Metric | Buy & Hold (SPY) | GMM Strategy | Impact |
| :--- | :---: | :---: | :---: |
| **Sharpe Ratio** | 0.61 | **0.73** | **+19.7% risk-adjusted return** |
| **Max Drawdown (MDD)** | -33.72% | **-15.64%** | **Downside risk cut by >53%** |
| **Average Exposure** | 100% | ~65–75% | Lower systematic beta / tail-risk protection |

---

## Core Quantitative Highlights

* **Zero Lookahead Leakage:** Features (Log Returns, 20d Realized Volatility, Normalized ATR) are standardized strictly via **expanding-window scaling** ($\min=60$ bars).
* **Information-Theoretic Tuning:** Regime cardinality ($K^* = 5$) is selected by minimizing the **Bayesian Information Criterion (BIC)** to prevent overfitting.
* **Probabilistic Allocation:** Avoids naive binary switches; computes continuous portfolio weights $w_t = \sum \tilde{P}_k \cdot \text{weight}_k$ using **EMA-smoothed posterior responsibilities** to suppress whipsaws.
* **Realistic Execution:** Weights are lagged ($t+1$) to ensure strict walk-forward validity.

---


# Install dependencies
pip install numpy pandas yfinance scikit-learn matplotlib

# Run pipeline
python gmm_regime_classifier.py
