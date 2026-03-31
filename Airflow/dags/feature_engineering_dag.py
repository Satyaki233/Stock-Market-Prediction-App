"""
Indian Stock Market - Feature Engineering DAG
Airflow 3.x (uses airflow.sdk imports)

Schedule: Daily at 5:00 PM IST (11:30 UTC) — 30 min after ingestion DAG (11:00 UTC)

Source tables  : price_history, fundamentals, dividends, splits, nse_symbol_registry
Output table   : stock_features

Features computed
─────────────────
Price / Returns  : daily_return, log_return, weekly_return, monthly_return
Moving averages  : sma_5/10/20/50/200, ema_12/26
Momentum         : macd, macd_signal, macd_hist, rsi_14
Volatility       : bb_upper/lower/width/pct, atr_14, rolling_std_5/20
Oscillators      : stoch_k, stoch_d
Volume           : volume_sma_20, volume_ratio
Price levels     : high_52w, low_52w, pct_from_52w_high, pct_from_52w_low
Lag features     : close_lag_1/2/5, return_lag_1/2/5
Fundamentals     : pe_ratio, pb_ratio, eps, revenue, net_income,
                   debt_to_equity, roe, market_cap  (forward-filled to each date)
Dividend events  : dividend_yield_ttm, days_since_dividend
Split events     : days_since_split
Registry meta    : sector, industry  (categorical, for model encoding)
Targets (labels) : target_return_1d, target_return_5d,
                   target_direction_1d (1=up / 0=down), target_direction_5d
"""

import logging

import pendulum
from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

log = logging.getLogger(__name__)

POSTGRES_CONN_ID = "stock_db_conn"


# ─────────────────────────────────────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────────────────────────────────────

@dag(
    dag_id="stock_feature_engineering",
    description="Compute ML features from raw stock data and store in stock_features",
    schedule="30 11 * * 1-5",          # 11:30 UTC = 5:00 PM IST, weekdays
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["stocks", "features", "ml", "india"],
    default_args={
        "retries": 2,
        "retry_delay": pendulum.duration(minutes=5),
        "owner": "data-team",
    },
)
def stock_feature_engineering():

    # ─────────────────────────────────────────
    # Task 1 — Validate / create output table
    # ─────────────────────────────────────────

    @task()
    def validate_feature_tables():
        """Create stock_features table and supporting indexes if they don't exist."""
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        hook.run("""
            CREATE TABLE IF NOT EXISTS stock_features (
                id                  SERIAL PRIMARY KEY,
                symbol              VARCHAR(30)  NOT NULL,
                date                DATE         NOT NULL,

                -- ── Returns ──────────────────────────────────────────────
                daily_return        NUMERIC(12,8),
                log_return          NUMERIC(12,8),
                weekly_return       NUMERIC(12,8),   -- 5-day
                monthly_return      NUMERIC(12,8),   -- 21-day

                -- ── Moving Averages ───────────────────────────────────────
                sma_5               NUMERIC(14,4),
                sma_10              NUMERIC(14,4),
                sma_20              NUMERIC(14,4),
                sma_50              NUMERIC(14,4),
                sma_200             NUMERIC(14,4),
                ema_12              NUMERIC(14,4),
                ema_26              NUMERIC(14,4),

                -- ── MACD ─────────────────────────────────────────────────
                macd                NUMERIC(14,6),
                macd_signal         NUMERIC(14,6),
                macd_hist           NUMERIC(14,6),

                -- ── RSI ──────────────────────────────────────────────────
                rsi_14              NUMERIC(8,4),

                -- ── Bollinger Bands (20-day, 2σ) ─────────────────────────
                bb_upper            NUMERIC(14,4),
                bb_lower            NUMERIC(14,4),
                bb_width            NUMERIC(12,6),
                bb_pct              NUMERIC(12,6),   -- %B oscillator

                -- ── ATR & Stochastic ──────────────────────────────────────
                atr_14              NUMERIC(14,4),
                stoch_k             NUMERIC(8,4),    -- Fast %K
                stoch_d             NUMERIC(8,4),    -- Slow %D (3-day SMA of K)

                -- ── Volume ───────────────────────────────────────────────
                volume_sma_20       NUMERIC(22,2),
                volume_ratio        NUMERIC(12,4),   -- volume / volume_sma_20

                -- ── Volatility ────────────────────────────────────────────
                rolling_std_5       NUMERIC(12,8),
                rolling_std_20      NUMERIC(12,8),

                -- ── 52-week price levels ──────────────────────────────────
                high_52w            NUMERIC(14,4),
                low_52w             NUMERIC(14,4),
                pct_from_52w_high   NUMERIC(12,6),
                pct_from_52w_low    NUMERIC(12,6),

                -- ── Lag features ──────────────────────────────────────────
                close_lag_1         NUMERIC(14,4),
                close_lag_2         NUMERIC(14,4),
                close_lag_5         NUMERIC(14,4),
                return_lag_1        NUMERIC(12,8),
                return_lag_2        NUMERIC(12,8),
                return_lag_5        NUMERIC(12,8),

                -- ── Fundamentals (forward-filled from latest available) ───
                pe_ratio            NUMERIC(12,4),
                pb_ratio            NUMERIC(12,4),
                eps                 NUMERIC(12,4),
                revenue             BIGINT,
                net_income          BIGINT,
                debt_to_equity      NUMERIC(12,4),
                roe                 NUMERIC(12,4),
                market_cap          BIGINT,

                -- ── Dividend features ─────────────────────────────────────
                dividend_yield_ttm  NUMERIC(12,8),   -- TTM dividends / close price
                days_since_dividend INTEGER,

                -- ── Split features ────────────────────────────────────────
                days_since_split    INTEGER,

                -- ── Registry meta (categorical) ───────────────────────────
                sector              VARCHAR(100),
                industry            VARCHAR(200),

                -- ── Target variables (supervised ML labels) ───────────────
                target_return_1d    NUMERIC(12,8),   -- next-day return
                target_return_5d    NUMERIC(12,8),   -- 5-day forward return
                target_direction_1d SMALLINT,        -- 1=up, 0=down (next day)
                target_direction_5d SMALLINT,        -- 1=up, 0=down (5 days out)

                -- ── Metadata ─────────────────────────────────────────────
                computed_at         TIMESTAMP DEFAULT NOW(),

                UNIQUE (symbol, date)
            );

            CREATE INDEX IF NOT EXISTS idx_sf_symbol_date
                ON stock_features(symbol, date DESC);
            CREATE INDEX IF NOT EXISTS idx_sf_date
                ON stock_features(date DESC);
            CREATE INDEX IF NOT EXISTS idx_sf_sector
                ON stock_features(sector);
        """)
        log.info("stock_features table validated / created.")
        return "tables_ready"

    # ─────────────────────────────────────────
    # Task 2 — Fetch symbol list
    # ─────────────────────────────────────────

    @task()
    def fetch_symbols(_: str) -> list:
        """
        Load active equity symbols from nse_symbol_registry.
        Also returns sector/industry metadata keyed by yf_symbol.
        Falls back to a small hardcoded list if the registry is unavailable.
        """
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        try:
            rows = hook.get_records("""
                SELECT yf_symbol, sector, industry
                FROM   nse_symbol_registry
                WHERE  symbol_type = 'equity'
                  AND  series      = 'EQ'
                  AND  is_active   = TRUE
                  AND  highlight   = 1
                ORDER BY symbol
            """)
            symbols = [
                {"yf_symbol": r[0], "sector": r[1], "industry": r[2]}
                for r in rows if r[0]
            ]
            if not symbols:
                raise ValueError("nse_symbol_registry returned 0 rows.")
            log.info(f"Loaded {len(symbols)} symbols from registry.")
            return symbols
        except Exception as e:
            log.warning(f"Registry unavailable ({e}). Using fallback list.")
            fallback = [
                "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
                "ICICIBANK.NS", "WIPRO.NS", "SBIN.NS", "TATAMOTORS.NS",
                "AXISBANK.NS", "LT.NS", "HINDUNILVR.NS", "ITC.NS",
                "BAJFINANCE.NS", "MARUTI.NS", "ADANIENT.NS", "NESTLEIND.NS",
            ]
            return [{"yf_symbol": s, "sector": None, "industry": None} for s in fallback]

    # ─────────────────────────────────────────
    # Task 3 — Compute and upsert features
    # ─────────────────────────────────────────

    @task()
    def compute_features(symbol_meta: list, dag_run_id: str):
        """
        For each equity symbol:
          1. Load full price_history from DB
          2. Compute all technical indicators (pure pandas / numpy)
          3. Forward-fill fundamentals onto every trading date (merge_asof)
          4. Compute dividend_yield_ttm and days_since_dividend (vectorised)
          5. Compute days_since_split (vectorised)
          6. Attach sector / industry from registry
          7. Compute target variables (1d & 5d forward returns / direction)
          8. Upsert results into stock_features via a per-symbol staging table
        """
        import numpy as np
        import pandas as pd
        from sqlalchemy import text

        hook   = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        engine = hook.get_sqlalchemy_engine()

        total_success = 0
        total_failed  = 0

        # All columns that land in stock_features (in insertion order)
        FEATURE_COLS = [
            "symbol", "date",
            "daily_return", "log_return", "weekly_return", "monthly_return",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26",
            "macd", "macd_signal", "macd_hist",
            "rsi_14",
            "bb_upper", "bb_lower", "bb_width", "bb_pct",
            "atr_14", "stoch_k", "stoch_d",
            "volume_sma_20", "volume_ratio",
            "rolling_std_5", "rolling_std_20",
            "high_52w", "low_52w", "pct_from_52w_high", "pct_from_52w_low",
            "close_lag_1", "close_lag_2", "close_lag_5",
            "return_lag_1", "return_lag_2", "return_lag_5",
            "pe_ratio", "pb_ratio", "eps", "revenue", "net_income",
            "debt_to_equity", "roe", "market_cap",
            "dividend_yield_ttm", "days_since_dividend",
            "days_since_split",
            "sector", "industry",
            "target_return_1d", "target_return_5d",
            "target_direction_1d", "target_direction_5d",
        ]

        for i, meta in enumerate(symbol_meta):
            symbol   = meta["yf_symbol"]
            sector   = meta.get("sector")
            industry = meta.get("industry")

            try:
                # ── 1. Load price history ─────────────────────────────────
                price_df = pd.read_sql(
                    """
                    SELECT date, open, high, low, close, adj_close, volume
                    FROM   price_history
                    WHERE  symbol = %(sym)s
                    ORDER  BY date ASC
                    """,
                    engine,
                    params={"sym": symbol},
                )

                if len(price_df) < 60:
                    log.warning(
                        f"[features] ({i+1}/{len(symbol_meta)}) "
                        f"{symbol}: only {len(price_df)} rows — skipping (need ≥60)"
                    )
                    continue

                price_df["date"] = pd.to_datetime(price_df["date"])
                price_df = price_df.sort_values("date").reset_index(drop=True)

                c = price_df["close"].astype(float)
                h = price_df["high"].astype(float)
                l = price_df["low"].astype(float)
                v = price_df["volume"].astype(float)

                df = price_df[["date"]].copy()
                df["symbol"] = symbol

                # ── 2. Technical indicators ───────────────────────────────

                # Returns
                df["daily_return"]   = c.pct_change()
                df["log_return"]     = np.log(c / c.shift(1))
                df["weekly_return"]  = c.pct_change(5)
                df["monthly_return"] = c.pct_change(21)

                # Moving averages
                df["sma_5"]   = c.rolling(5).mean()
                df["sma_10"]  = c.rolling(10).mean()
                df["sma_20"]  = c.rolling(20).mean()
                df["sma_50"]  = c.rolling(50).mean()
                df["sma_200"] = c.rolling(200).mean()
                df["ema_12"]  = c.ewm(span=12, adjust=False).mean()
                df["ema_26"]  = c.ewm(span=26, adjust=False).mean()

                # MACD (12/26/9)
                df["macd"]        = df["ema_12"] - df["ema_26"]
                df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
                df["macd_hist"]   = df["macd"] - df["macd_signal"]

                # RSI-14 (Wilder's smoothing via EWM com=13)
                delta    = c.diff()
                gain     = delta.clip(lower=0)
                loss     = (-delta).clip(lower=0)
                avg_gain = gain.ewm(com=13, adjust=False).mean()
                avg_loss = loss.ewm(com=13, adjust=False).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                df["rsi_14"] = 100 - (100 / (1 + rs))

                # Bollinger Bands (20, 2σ)
                bb_mid         = c.rolling(20).mean()
                bb_std         = c.rolling(20).std()
                df["bb_upper"] = bb_mid + 2 * bb_std
                df["bb_lower"] = bb_mid - 2 * bb_std
                bb_range       = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
                df["bb_width"] = bb_range / bb_mid
                df["bb_pct"]   = (c - df["bb_lower"]) / bb_range   # %B

                # ATR-14 (Wilder's smoothing)
                tr = pd.concat([
                    h - l,
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs(),
                ], axis=1).max(axis=1)
                df["atr_14"] = tr.ewm(com=13, adjust=False).mean()

                # Stochastic Oscillator (14, 3)
                low14        = l.rolling(14).min()
                high14       = h.rolling(14).max()
                stoch_range  = (high14 - low14).replace(0, np.nan)
                df["stoch_k"] = 100 * (c - low14) / stoch_range
                df["stoch_d"] = df["stoch_k"].rolling(3).mean()

                # Volume
                df["volume_sma_20"] = v.rolling(20).mean()
                df["volume_ratio"]  = v / df["volume_sma_20"].replace(0, np.nan)

                # Rolling volatility
                df["rolling_std_5"]  = df["daily_return"].rolling(5).std()
                df["rolling_std_20"] = df["daily_return"].rolling(20).std()

                # 52-week (252 trading-day) high / low
                df["high_52w"]          = h.rolling(252).max()
                df["low_52w"]           = l.rolling(252).min()
                df["pct_from_52w_high"] = (c - df["high_52w"]) / df["high_52w"].replace(0, np.nan)
                df["pct_from_52w_low"]  = (c - df["low_52w"])  / df["low_52w"].replace(0, np.nan)

                # Lag features
                df["close_lag_1"]  = c.shift(1)
                df["close_lag_2"]  = c.shift(2)
                df["close_lag_5"]  = c.shift(5)
                df["return_lag_1"] = df["daily_return"].shift(1)
                df["return_lag_2"] = df["daily_return"].shift(2)
                df["return_lag_5"] = df["daily_return"].shift(5)

                # Target variables (look-ahead — only valid for historical rows)
                future_close_1d       = c.shift(-1)
                future_close_5d       = c.shift(-5)
                df["target_return_1d"]    = (future_close_1d - c) / c.replace(0, np.nan)
                df["target_return_5d"]    = (future_close_5d - c) / c.replace(0, np.nan)
                df["target_direction_1d"] = (df["target_return_1d"] > 0).astype("Int8")
                df["target_direction_5d"] = (df["target_return_5d"] > 0).astype("Int8")

                # ── 3. Fundamentals — forward-fill to each trading date ───
                fund_df = pd.read_sql(
                    """
                    SELECT as_of_date, pe_ratio, pb_ratio, eps,
                           revenue, net_income, debt_to_equity, roe, market_cap
                    FROM   fundamentals
                    WHERE  symbol = %(sym)s
                    ORDER  BY as_of_date ASC
                    """,
                    engine,
                    params={"sym": symbol},
                )

                if not fund_df.empty:
                    fund_df["as_of_date"] = pd.to_datetime(fund_df["as_of_date"])
                    fund_df = fund_df.rename(columns={"as_of_date": "date"})
                    # merge_asof: for each trading date, use the most recent
                    # fundamental snapshot that is ≤ that date
                    df = pd.merge_asof(
                        df.sort_values("date"),
                        fund_df.sort_values("date"),
                        on="date",
                        direction="backward",
                    )
                else:
                    for col in ["pe_ratio", "pb_ratio", "eps", "revenue",
                                "net_income", "debt_to_equity", "roe", "market_cap"]:
                        df[col] = None

                # ── 4. Dividend features (vectorised) ────────────────────
                div_df = pd.read_sql(
                    """
                    SELECT ex_date, amount
                    FROM   dividends
                    WHERE  symbol = %(sym)s
                    ORDER  BY ex_date ASC
                    """,
                    engine,
                    params={"sym": symbol},
                )

                price_dates = pd.to_datetime(df["date"])

                if not div_df.empty:
                    div_df["ex_date"] = pd.to_datetime(div_df["ex_date"])

                    # TTM dividend yield — vectorised rolling sum
                    # Reindex dividend amounts onto every trading date (0 on non-event days)
                    div_series = (
                        div_df.set_index("ex_date")["amount"]
                        .reindex(price_dates)
                        .fillna(0.0)
                    )
                    div_series.index = price_dates
                    ttm_div_sum = div_series.rolling(252, min_periods=1).sum()
                    df["dividend_yield_ttm"] = (
                        ttm_div_sum.values / c.replace(0, np.nan).values
                    )

                    # Days since last dividend — vectorised with searchsorted
                    sorted_div_dates = div_df["ex_date"].sort_values().values
                    price_dates_arr  = price_dates.values

                    def _days_since(price_arr, event_arr):
                        """Return number of calendar days since the most recent event."""
                        idx = np.searchsorted(event_arr, price_arr, side="right") - 1
                        result = np.where(
                            idx >= 0,
                            (price_arr - event_arr[np.clip(idx, 0, len(event_arr) - 1)])
                            .astype("timedelta64[D]").astype(float),
                            np.nan,
                        )
                        # Mask rows where idx < 0 (no prior event)
                        result[idx < 0] = np.nan
                        return result

                    df["days_since_dividend"] = _days_since(
                        price_dates_arr, sorted_div_dates
                    )
                else:
                    df["dividend_yield_ttm"]  = None
                    df["days_since_dividend"] = None

                # ── 5. Split features (vectorised) ────────────────────────
                split_df = pd.read_sql(
                    """
                    SELECT split_date
                    FROM   splits
                    WHERE  symbol = %(sym)s
                    ORDER  BY split_date ASC
                    """,
                    engine,
                    params={"sym": symbol},
                )

                if not split_df.empty:
                    split_df["split_date"] = pd.to_datetime(split_df["split_date"])
                    sorted_split_dates = split_df["split_date"].sort_values().values
                    df["days_since_split"] = _days_since(
                        price_dates_arr, sorted_split_dates
                    )
                else:
                    df["days_since_split"] = None

                # ── 6. Registry metadata ──────────────────────────────────
                df["sector"]   = sector
                df["industry"] = industry

                # ── 7. Final cleanup & upsert ─────────────────────────────
                df["date"] = df["date"].dt.date

                # Replace inf values before any type coercion
                df.replace([np.inf, -np.inf], np.nan, inplace=True)

                # Convert nullable Int8 columns to plain Python int/None
                # so psycopg2 writes NULL (not NaN string) to SMALLINT columns
                for col in ("target_direction_1d", "target_direction_5d"):
                    if col in df.columns:
                        df[col] = df[col].astype(object).where(df[col].notna(), None)

                # Keep only the columns that belong in the output table
                available = [col for col in FEATURE_COLS if col in df.columns]
                df_out = df[available].copy()

                # Coerce object-dtype numeric columns to float64.
                # Columns set to None (e.g. fundamentals missing for a symbol)
                # end up as object dtype — pandas to_sql maps object → TEXT,
                # which causes a DatatypeMismatch against NUMERIC in stock_features.
                _non_numeric = {"symbol", "date", "sector", "industry",
                                "target_direction_1d", "target_direction_5d"}
                for col in available:
                    if col not in _non_numeric and df_out[col].dtype == object:
                        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

                # Stage → upsert → drop staging
                # Use a lowercase name so pandas to_sql and the raw SQL query
                # both refer to the same identifier (PostgreSQL lowercases unquoted names).
                # Critically: pass `conn` (not `engine`) to to_sql so the staging
                # table creation and the INSERT run on the SAME connection — this
                # guarantees the table is visible when the upsert executes.
                staging     = f"_staging_feat_{symbol.replace('.', '_').replace('^', '').replace('=', '_')}_tmp".lower()
                upsert_cols = [col for col in available if col not in ("symbol", "date")]
                set_clause  = ",\n                            ".join(
                    f"{col} = EXCLUDED.{col}" for col in upsert_cols
                )

                with engine.begin() as conn:
                    df_out.to_sql(
                        staging, conn,
                        if_exists="replace",
                        index=False,
                        method="multi",
                        chunksize=2000,
                    )
                    result = conn.execute(text(f"""
                        INSERT INTO stock_features ({", ".join(available)})
                        SELECT {", ".join(available)}
                        FROM   {staging}
                        ON CONFLICT (symbol, date) DO UPDATE SET
                            {set_clause},
                            computed_at = NOW()
                    """))
                    upserted = result.rowcount
                    conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))

                hook.insert_rows(
                    table="ingestion_log",
                    rows=[(symbol, "features", "success", upserted, None, dag_run_id)],
                    target_fields=["symbol", "data_type", "status",
                                   "rows_affected", "error_msg", "dag_run_id"],
                )
                log.info(
                    f"[features] ({i+1}/{len(symbol_meta)}) "
                    f"{symbol}: {upserted} rows upserted"
                )
                total_success += 1

            except Exception as e:
                hook.insert_rows(
                    table="ingestion_log",
                    rows=[(symbol, "features", "failed", 0, str(e)[:200], dag_run_id)],
                    target_fields=["symbol", "data_type", "status",
                                   "rows_affected", "error_msg", "dag_run_id"],
                )
                log.error(f"[features] ({i+1}/{len(symbol_meta)}) {symbol} FAILED: {e}")
                total_failed += 1

        log.info(
            f"[features] Completed. "
            f"Success: {total_success}, Failed: {total_failed}, "
            f"Total: {len(symbol_meta)}"
        )

    # ─────────────────────────────────────────
    # Task 4 — Summary report
    # ─────────────────────────────────────────

    @task()
    def feature_summary(dag_run_id: str):
        """Query ingestion_log and print a feature-engineering run summary."""
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        rows = hook.get_records("""
            SELECT status, COUNT(*) AS symbols, SUM(rows_affected) AS total_rows
            FROM   ingestion_log
            WHERE  dag_run_id = %s
              AND  data_type  = 'features'
            GROUP  BY status
            ORDER  BY status
        """, parameters=(dag_run_id,))

        total_symbols = hook.get_first("""
            SELECT COUNT(DISTINCT symbol) FROM stock_features
        """)[0]

        log.info("=" * 55)
        log.info(f"FEATURE ENGINEERING SUMMARY — Run: {dag_run_id}")
        log.info("=" * 55)
        for status, symbols, total_rows in rows:
            log.info(f"  {status:<10} | symbols: {symbols:<5} | rows: {total_rows}")
        log.info(f"  Total symbols in stock_features: {total_symbols}")
        log.info("=" * 55)

    # ─────────────────────────────────────────
    # Task flow
    # ─────────────────────────────────────────

    dag_run_id    = "{{ run_id }}"
    tables_ready  = validate_feature_tables()
    symbol_meta   = fetch_symbols(tables_ready)
    features_done = compute_features(symbol_meta=symbol_meta, dag_run_id=dag_run_id)
    features_done >> feature_summary(dag_run_id=dag_run_id)


# Register the DAG
stock_feature_engineering()
