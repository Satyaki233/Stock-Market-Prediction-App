# Feature Engineering — Stock Market Prediction

**DAG:** `stock_feature_engineering`
**Output table:** `stock_features`
**Schedule:** Daily at 5:00 PM IST (11:30 UTC), weekdays only
**Depends on:** `indian_stock_data_ingestion` (runs at 4:30 PM IST)

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Source Tables](#2-source-tables)
3. [Returns](#3-returns)
4. [Moving Averages](#4-moving-averages)
5. [MACD](#5-macd)
6. [RSI](#6-rsi)
7. [Bollinger Bands](#7-bollinger-bands)
8. [ATR — Average True Range](#8-atr--average-true-range)
9. [Stochastic Oscillator](#9-stochastic-oscillator)
10. [Volume Features](#10-volume-features)
11. [Volatility](#11-volatility)
12. [52-Week Price Levels](#12-52-week-price-levels)
13. [Lag Features](#13-lag-features)
14. [Fundamentals](#14-fundamentals)
15. [Dividend Features](#15-dividend-features)
16. [Split Features](#16-split-features)
17. [Registry Metadata](#17-registry-metadata)
18. [Target Variables](#18-target-variables)
19. [Output Schema](#19-output-schema)
20. [Design Decisions](#20-design-decisions)

---

## 1. Pipeline Overview

```
price_history  ──┐
fundamentals   ──┤
dividends      ──┼──► compute_features ──► stock_features
splits         ──┤
nse_symbol_    ──┘
  registry
```

For each equity symbol the DAG:

1. Loads the full price history from `price_history`
2. Computes all technical indicators in-memory with **pandas / numpy** (no TA-Lib)
3. Forward-fills the latest fundamental snapshot onto every trading date (`merge_asof`)
4. Vectorises dividend and split event lookups with `np.searchsorted`
5. Attaches sector / industry from `nse_symbol_registry`
6. Computes look-ahead target labels for supervised ML
7. Upserts the result into `stock_features` via a per-symbol staging table

All calculations are **idempotent** — re-running any day overwrites existing rows.

---

## 2. Source Tables

| Table | Key columns used |
|---|---|
| `price_history` | `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume` |
| `fundamentals` | `as_of_date`, `pe_ratio`, `pb_ratio`, `eps`, `revenue`, `net_income`, `debt_to_equity`, `roe`, `market_cap` |
| `dividends` | `ex_date`, `amount` |
| `splits` | `split_date` |
| `nse_symbol_registry` | `yf_symbol`, `sector`, `industry` |

---

## 3. Returns

Returns measure how much the price changed over a given period.

### Daily Return
```
daily_return = (close_t - close_{t-1}) / close_{t-1}
             = close.pct_change()
```
Simple percentage change from the previous trading day.

### Log Return
```
log_return = ln(close_t / close_{t-1})
           = log(close / close.shift(1))
```
Log returns are additive over time and approximately normally distributed — preferred for statistical modelling.

### Weekly Return (5-day)
```
weekly_return = (close_t - close_{t-5}) / close_{t-5}
              = close.pct_change(5)
```

### Monthly Return (21-day)
```
monthly_return = (close_t - close_{t-21}) / close_{t-21}
               = close.pct_change(21)
```

---

## 4. Moving Averages

Moving averages smooth price data and identify trend direction.

### Simple Moving Average (SMA)
```
SMA_n(t) = mean(close_{t-n+1}, ..., close_t)
         = close.rolling(n).mean()
```

Computed for windows: **5, 10, 20, 50, 200** days.

| Feature | Window | Common use |
|---|---|---|
| `sma_5` | 5 days | Very short-term noise filter |
| `sma_10` | 10 days | Short-term trend |
| `sma_20` | 20 days | Base for Bollinger Bands |
| `sma_50` | 50 days | Medium-term trend |
| `sma_200` | 200 days | Long-term bull/bear signal |

**Interpretation:** When `close > sma_200` the stock is in a long-term uptrend. A `sma_50` crossing above `sma_200` is the classic "Golden Cross" buy signal.

### Exponential Moving Average (EMA)
```
EMA_n(t) = close_t * α  +  EMA_n(t-1) * (1 - α)
         where  α = 2 / (n + 1)
         = close.ewm(span=n, adjust=False).mean()
```

Computed for spans: **12** and **26** days (inputs to MACD).

EMA gives more weight to recent prices than SMA — it reacts faster to price changes.

---

## 5. MACD

**Moving Average Convergence Divergence** — momentum indicator.

```
MACD        = EMA_12 - EMA_26
MACD_signal = EMA_9(MACD)          [9-period EMA of the MACD line]
MACD_hist   = MACD - MACD_signal   [histogram / divergence bar]
```

| Feature | Meaning |
|---|---|
| `macd` | Difference between fast (12) and slow (26) EMA |
| `macd_signal` | Smoothed trigger line |
| `macd_hist` | Gap between MACD and signal — shows momentum strength |

**Interpretation:**
- `macd > 0` → short-term trend is up relative to long-term
- `macd_hist > 0` and growing → accelerating bullish momentum
- Signal crossover (MACD crosses above signal) → buy signal

---

## 6. RSI

**Relative Strength Index** — momentum oscillator bounded 0–100.

```
delta    = close.diff()
gain     = delta.clip(lower=0)            [positive changes only]
loss     = (-delta).clip(lower=0)         [absolute negative changes only]

avg_gain = gain.ewm(com=13, adjust=False).mean()    [Wilder's smoothing: α = 1/14]
avg_loss = loss.ewm(com=13, adjust=False).mean()

RS       = avg_gain / avg_loss
RSI_14   = 100 - (100 / (1 + RS))
```

**Wilder's smoothing** uses `com = period - 1 = 13`, equivalent to `α = 1/14`.

| Value | Interpretation |
|---|---|
| RSI > 70 | Overbought — potential reversal downward |
| RSI < 30 | Oversold — potential reversal upward |
| RSI = 50 | Neutral; trend transition zone |

---

## 7. Bollinger Bands

**Bollinger Bands** place an envelope around price using a 20-day SMA ± 2 standard deviations.

```
bb_mid   = SMA_20
bb_std   = close.rolling(20).std()

bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_width = (bb_upper - bb_lower) / bb_mid     [normalised band width]
bb_pct   = (close - bb_lower) / (bb_upper - bb_lower)   [%B position]
```

| Feature | Range | Interpretation |
|---|---|---|
| `bb_upper` | price units | Resistance / overbought boundary |
| `bb_lower` | price units | Support / oversold boundary |
| `bb_width` | 0 → ∞ | Volatility; narrow bands → potential breakout |
| `bb_pct` | 0 to 1 | 1 = at upper band, 0 = at lower band, 0.5 = at mid |

---

## 8. ATR — Average True Range

**ATR** measures market volatility as the average of the True Range over 14 days.

```
True Range (TR) = max(
    high - low,                    [today's range]
    |high - close_{t-1}|,          [gap-up scenario]
    |low  - close_{t-1}|           [gap-down scenario]
)

ATR_14 = TR.ewm(com=13, adjust=False).mean()    [Wilder's smoothing]
```

**Interpretation:** A rising ATR means increasing volatility. ATR is used to:
- Size stop-losses (e.g. 2× ATR below entry)
- Filter signals — only trade when ATR is above a threshold

---

## 9. Stochastic Oscillator

**Stochastic (14, 3)** compares closing price to the 14-day high-low range.

```
low14   = low.rolling(14).min()
high14  = high.rolling(14).max()

stoch_k = 100 * (close - low14) / (high14 - low14)    [Fast %K]
stoch_d = stoch_k.rolling(3).mean()                   [Slow %D — signal line]
```

| Value | Interpretation |
|---|---|
| stoch_k > 80 | Overbought |
| stoch_k < 20 | Oversold |
| K crosses above D | Buy signal |
| K crosses below D | Sell signal |

---

## 10. Volume Features

Volume confirms the strength of price moves.

```
volume_sma_20 = volume.rolling(20).mean()
volume_ratio  = volume / volume_sma_20
```

| Feature | Interpretation |
|---|---|
| `volume_sma_20` | Baseline average daily volume |
| `volume_ratio` | > 1 means above-average activity; > 2 signals unusual interest |

**Use in ML:** Volume spikes often precede or confirm breakouts. A price move on `volume_ratio > 2` is more reliable than one on low volume.

---

## 11. Volatility

Rolling standard deviation of daily returns — measures how much returns vary.

```
rolling_std_5  = daily_return.rolling(5).std()
rolling_std_20 = daily_return.rolling(20).std()
```

| Feature | Annualised equivalent | Use |
|---|---|---|
| `rolling_std_5` | `× sqrt(252)` | Short-term risk; sensitive to recent events |
| `rolling_std_20` | `× sqrt(252)` | Medium-term historical volatility |

---

## 12. 52-Week Price Levels

Measures where today's price sits relative to the annual range.

```
high_52w          = high.rolling(252).max()
low_52w           = low.rolling(252).min()

pct_from_52w_high = (close - high_52w) / high_52w    [always ≤ 0]
pct_from_52w_low  = (close - low_52w)  / low_52w     [always ≥ 0]
```

**Example:** `pct_from_52w_high = -0.20` means the stock is 20% below its 52-week high — a potential mean-reversion setup.

---

## 13. Lag Features

Lag features give the model direct access to recent historical values without the model needing to learn the lookback internally.

```
close_lag_1  = close.shift(1)    [yesterday's close]
close_lag_2  = close.shift(2)
close_lag_5  = close.shift(5)    [close 5 trading days ago]

return_lag_1 = daily_return.shift(1)
return_lag_2 = daily_return.shift(2)
return_lag_5 = daily_return.shift(5)
```

These encode short-term price memory and are especially useful for models that don't have recurrence (e.g. gradient boosting, linear regression).

---

## 14. Fundamentals

Fundamental data is collected **once per day** by the ingestion DAG (current snapshot from yfinance). The feature DAG forward-fills each snapshot onto every subsequent trading date using `pandas.merge_asof`:

```python
pd.merge_asof(
    price_df.sort_values("date"),
    fund_df.sort_values("as_of_date"),   # renamed to "date"
    on="date",
    direction="backward",               # use most recent snapshot ≤ trade date
)
```

This ensures that on any given date the model only sees fundamental data that would have been available at that time — **no look-ahead bias**.

| Feature | Description |
|---|---|
| `pe_ratio` | Price / Trailing Earnings |
| `pb_ratio` | Price / Book Value |
| `eps` | Trailing Earnings Per Share |
| `revenue` | Total Revenue (TTM) |
| `net_income` | Net Income (TTM) |
| `debt_to_equity` | Total Debt / Shareholders Equity |
| `roe` | Return on Equity |
| `market_cap` | Market Capitalisation |

---

## 15. Dividend Features

### TTM Dividend Yield

```
# Build a daily series: dividend amount on ex-date, 0 elsewhere
div_series      = div_df.set_index("ex_date")["amount"]
                         .reindex(price_dates)
                         .fillna(0.0)

ttm_div_sum     = div_series.rolling(252, min_periods=1).sum()

dividend_yield_ttm = ttm_div_sum / close
```

Rolling 252-day sum approximates the **trailing twelve-month** dividend. Dividing by close gives the current annualised yield.

### Days Since Last Dividend

```python
idx = np.searchsorted(sorted_div_dates, price_date, side="right") - 1
days_since_dividend = (price_date - sorted_div_dates[idx]).days   # if idx >= 0
```

Uses binary search (`np.searchsorted`) for O(log n) lookup across all trading dates — fully vectorised, no `.apply()`.

**Use in ML:** Stocks tend to rise before the ex-dividend date and drop by approximately the dividend amount on the ex-date. `days_since_dividend` encodes proximity to this event.

---

## 16. Split Features

### Days Since Last Split

Identical vectorised approach to dividends:

```python
idx = np.searchsorted(sorted_split_dates, price_date, side="right") - 1
days_since_split = (price_date - sorted_split_dates[idx]).days
```

Stock splits often coincide with periods of strong price appreciation and increased retail activity. The feature helps the model learn patterns around split events.

---

## 17. Registry Metadata

`sector` and `industry` are pulled from `nse_symbol_registry` and stored as **raw strings** in `stock_features`.

At model training time these should be encoded, e.g.:
- **Label encoding** for tree-based models (XGBoost, LightGBM)
- **One-hot encoding** for linear / neural models

Sector-level features allow the model to learn that, for example, momentum signals work differently in IT vs. FMCG stocks.

---

## 18. Target Variables

These are **supervised labels** — the correct answer the model tries to predict. They use look-ahead data and are therefore only valid for historical (non-latest) rows.

```
future_close_1d     = close.shift(-1)
future_close_5d     = close.shift(-5)

target_return_1d    = (future_close_1d - close) / close
target_return_5d    = (future_close_5d - close) / close

target_direction_1d = 1  if  target_return_1d > 0  else  0
target_direction_5d = 1  if  target_return_5d > 0  else  0
```

| Target | Type | Task |
|---|---|---|
| `target_return_1d` | float | Regression — predict exact next-day return |
| `target_return_5d` | float | Regression — predict 5-day forward return |
| `target_direction_1d` | 0 / 1 | Classification — predict up or down next day |
| `target_direction_5d` | 0 / 1 | Classification — predict up or down in 5 days |

> **Note:** The most recent rows (last 1 day for `_1d` targets, last 5 days for `_5d` targets) will have `NULL` targets because the future price is not yet available. These rows are excluded during model training but are exactly what the model predicts on in production.

---

## 19. Output Schema

Full `stock_features` table schema:

```sql
stock_features (
    id                  SERIAL PRIMARY KEY,
    symbol              VARCHAR(30)  NOT NULL,
    date                DATE         NOT NULL,

    -- Returns
    daily_return        NUMERIC(12,8),
    log_return          NUMERIC(12,8),
    weekly_return       NUMERIC(12,8),
    monthly_return      NUMERIC(12,8),

    -- Moving Averages
    sma_5               NUMERIC(14,4),
    sma_10              NUMERIC(14,4),
    sma_20              NUMERIC(14,4),
    sma_50              NUMERIC(14,4),
    sma_200             NUMERIC(14,4),
    ema_12              NUMERIC(14,4),
    ema_26              NUMERIC(14,4),

    -- MACD
    macd                NUMERIC(14,6),
    macd_signal         NUMERIC(14,6),
    macd_hist           NUMERIC(14,6),

    -- RSI
    rsi_14              NUMERIC(8,4),

    -- Bollinger Bands
    bb_upper            NUMERIC(14,4),
    bb_lower            NUMERIC(14,4),
    bb_width            NUMERIC(12,6),
    bb_pct              NUMERIC(12,6),

    -- ATR & Stochastic
    atr_14              NUMERIC(14,4),
    stoch_k             NUMERIC(8,4),
    stoch_d             NUMERIC(8,4),

    -- Volume
    volume_sma_20       NUMERIC(22,2),
    volume_ratio        NUMERIC(12,4),

    -- Volatility
    rolling_std_5       NUMERIC(12,8),
    rolling_std_20      NUMERIC(12,8),

    -- 52-week levels
    high_52w            NUMERIC(14,4),
    low_52w             NUMERIC(14,4),
    pct_from_52w_high   NUMERIC(12,6),
    pct_from_52w_low    NUMERIC(12,6),

    -- Lag features
    close_lag_1         NUMERIC(14,4),
    close_lag_2         NUMERIC(14,4),
    close_lag_5         NUMERIC(14,4),
    return_lag_1        NUMERIC(12,8),
    return_lag_2        NUMERIC(12,8),
    return_lag_5        NUMERIC(12,8),

    -- Fundamentals
    pe_ratio            NUMERIC(12,4),
    pb_ratio            NUMERIC(12,4),
    eps                 NUMERIC(12,4),
    revenue             BIGINT,
    net_income          BIGINT,
    debt_to_equity      NUMERIC(12,4),
    roe                 NUMERIC(12,4),
    market_cap          BIGINT,

    -- Dividend
    dividend_yield_ttm  NUMERIC(12,8),
    days_since_dividend INTEGER,

    -- Split
    days_since_split    INTEGER,

    -- Registry metadata
    sector              VARCHAR(100),
    industry            VARCHAR(200),

    -- Targets
    target_return_1d    NUMERIC(12,8),
    target_return_5d    NUMERIC(12,8),
    target_direction_1d SMALLINT,
    target_direction_5d SMALLINT,

    -- Meta
    computed_at         TIMESTAMP DEFAULT NOW(),

    UNIQUE (symbol, date)
)
```

---

## 20. Design Decisions

### No TA-Lib dependency
All indicators are implemented with pure pandas / numpy. This avoids a C-extension build dependency inside Docker and makes the code fully portable.

### Wilder's smoothing for RSI and ATR
Both RSI and ATR use **Wilder's smoothing** (EWM with `com = period - 1`), which is the original specification. This differs from a simple EMA (`span = period`) and matches charting platforms like TradingView.

### Vectorised event lookups
Dividend and split `days_since` calculations use `np.searchsorted` instead of row-wise `.apply()`. This gives O(n log k) time complexity (n = trading days, k = events) vs O(n·k) for a naive loop — important when processing 2500+ symbols with 10 years of daily data each.

### Forward-fill fundamentals with `merge_asof`
`pandas.merge_asof` with `direction="backward"` ensures each trading date only sees the most recent fundamental snapshot that has already been published — no look-ahead bias.

### Staging table upsert pattern
Each symbol's features are written to a temporary `_staging_feat_<symbol>_tmp` table, then upserted into `stock_features` in a single SQL statement, then the staging table is dropped. This is the same pattern used by the ingestion DAG and provides:
- Atomic per-symbol updates
- No partial writes if the task fails mid-symbol
- Conflict resolution (re-runs are safe)

### NULL handling
- `np.inf` / `-np.inf` values (from division by zero in ratio calculations) are replaced with `None` before writing
- Nullable `Int8` target direction columns are cast to Python `object` type so pandas writes `NULL` (not `NaN` string) to Postgres
- The last 1–5 rows per symbol will always have `NULL` targets — this is expected and correct
