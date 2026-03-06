"""
Indian Stock Market - Data Ingestion DAG
Airflow 3.x (uses airflow.sdk imports)

Schedule: Daily at 4:30 PM IST (11:00 UTC) — 1 hour after market close
Covers:
  - OHLCV price history (10 years, bulk insert via pandas to_sql)
  - Dividends
  - Stock splits
  - Fundamentals
  - Ingestion logs

Symbols are sourced from the nse_symbol_registry table (populated by the
nse_symbol_registry DAG). Falls back to a hardcoded list if the registry
is not yet available.
"""

import logging

import pendulum
from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

log = logging.getLogger(__name__)

POSTGRES_CONN_ID = "stock_db_conn"

# Fallback symbol lists used when nse_symbol_registry is empty / missing
_FALLBACK_EQUITIES = [
    "RELIANCE.NS", "TCS.NS",      "INFY.NS",      "HDFCBANK.NS",
    "ICICIBANK.NS", "WIPRO.NS",   "SBIN.NS",       "TATAMOTORS.NS",
    "AXISBANK.NS",  "LT.NS",      "HINDUNILVR.NS", "ITC.NS",
    "BAJFINANCE.NS","MARUTI.NS",  "ADANIENT.NS",   "NESTLEIND.NS",
]

_FALLBACK_INDICES = [
    "^NSEI", "^BSESN", "^INDIAVIX", "INR=X",
    "^CNXIT", "^NSEBANK", "^CNXPHARMA", "^CNXAUTO", "^CNXFMCG",
]

IST = pendulum.timezone("Asia/Kolkata")


# ─────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────

@dag(
    dag_id="indian_stock_data_ingestion",
    description="Fetch Indian stock data from yfinance and store in PostgreSQL",
    schedule="00 11 * * 1-5",          # 11:00 UTC = 4:30 PM IST, weekdays only
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["stocks", "ingestion", "india"],
    default_args={
        "retries": 3,
        "retry_delay": pendulum.duration(minutes=5),
        "owner": "data-team",
    },
)
def indian_stock_data_ingestion():

    # ─────────────────────────────────────────
    # Task 1: Validate DB tables exist
    # ─────────────────────────────────────────

    @task()
    def validate_db_tables():
        """
        Ensure all required tables exist before ingestion starts.
        Creates them if missing — safe for first run.
        """
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        ddl = """
              CREATE TABLE IF NOT EXISTS ingestion_delta_table (
                id            SERIAL PRIMARY KEY,
                symbol        VARCHAR(30)  NOT NULL,
                data_type     VARCHAR(30)  NOT NULL,  -- 'price', 'dividend', 'split', 'fundamental'
                data_from     DATE,                   -- earliest date of actual data fetched
                data_to       DATE,                   -- latest date of actual data fetched
                start_time    TIMESTAMP    NOT NULL,  -- when this symbol's ingestion started
                end_time      TIMESTAMP,              -- when it completed (NULL if crashed)
                status        VARCHAR(20)  NOT NULL,  -- 'success', 'failed', 'skipped'
                record_count  INT          DEFAULT 0,
                error_msg     TEXT,
                dag_run_id    VARCHAR(100),
                created_at    TIMESTAMP    DEFAULT NOW(),
                UNIQUE (symbol, data_type, start_time)
            );

            CREATE TABLE IF NOT EXISTS price_history (
                id          SERIAL PRIMARY KEY,
                symbol      VARCHAR(30) NOT NULL,
                date        DATE NOT NULL,
                open        NUMERIC(12,4),
                high        NUMERIC(12,4),
                low         NUMERIC(12,4),
                close       NUMERIC(12,4),
                adj_close   NUMERIC(12,4),
                volume      BIGINT,
                ingested_at TIMESTAMP DEFAULT NOW(),
                UNIQUE (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS dividends (
                id          SERIAL PRIMARY KEY,
                symbol      VARCHAR(30) NOT NULL,
                ex_date     DATE NOT NULL,
                amount      NUMERIC(10,4),
                ingested_at TIMESTAMP DEFAULT NOW(),
                UNIQUE (symbol, ex_date)
            );

            CREATE TABLE IF NOT EXISTS splits (
                id          SERIAL PRIMARY KEY,
                symbol      VARCHAR(30) NOT NULL,
                split_date  DATE NOT NULL,
                ratio       NUMERIC(10,4),
                ingested_at TIMESTAMP DEFAULT NOW(),
                UNIQUE (symbol, split_date)
            );

            CREATE TABLE IF NOT EXISTS fundamentals (
                id              SERIAL PRIMARY KEY,
                symbol          VARCHAR(30) NOT NULL,
                as_of_date      DATE NOT NULL,
                pe_ratio        NUMERIC(10,4),
                pb_ratio        NUMERIC(10,4),
                eps             NUMERIC(10,4),
                revenue         BIGINT,
                net_income      BIGINT,
                debt_to_equity  NUMERIC(10,4),
                roe             NUMERIC(10,4),
                market_cap      BIGINT,
                ingested_at     TIMESTAMP DEFAULT NOW(),
                UNIQUE (symbol, as_of_date)
            );

            CREATE TABLE IF NOT EXISTS ingestion_log (
                id            SERIAL PRIMARY KEY,
                symbol        VARCHAR(30),
                data_type     VARCHAR(30),
                status        VARCHAR(10),
                rows_affected INT DEFAULT 0,
                error_msg     TEXT,
                dag_run_id    VARCHAR(100),
                ran_at        TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_price_symbol_date
                ON price_history(symbol, date DESC);
            CREATE INDEX IF NOT EXISTS idx_price_date
                ON price_history(date DESC);
            CREATE INDEX IF NOT EXISTS idx_fund_symbol
                ON fundamentals(symbol, as_of_date DESC);
        """
        hook.run(ddl)
        log.info("DB tables validated / created successfully.")
        return "tables_ready"

    # ─────────────────────────────────────────
    # Task 2: Fetch symbols from nse_symbol_registry
    # ─────────────────────────────────────────

    @task()
    def fetch_symbols_from_db(_: str) -> dict:
        """
        Query nse_symbol_registry for all active equity and index yf_symbols.
        - Equities: symbol_type='equity', series='EQ', is_active=TRUE
        - Indices:  symbol_type='index',  is_active=TRUE

        Falls back to hardcoded lists if the registry table is missing or empty.
        """
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        try:
            equity_rows = hook.get_records("""
                SELECT yf_symbol
                FROM nse_symbol_registry
                WHERE symbol_type = 'equity'
                  AND series      = 'EQ'
                  AND is_active   = TRUE
                  AND highlight   = 1
                ORDER BY symbol
            """)

            index_rows = hook.get_records("""
                SELECT yf_symbol
                FROM nse_symbol_registry
                WHERE symbol_type = 'index'
                  AND is_active   = TRUE
                  AND highlight   = 1
                ORDER BY symbol
            """)

            equities = [r[0] for r in equity_rows if r[0]]
            indices  = [r[0] for r in index_rows  if r[0]]

            if not equities and not indices:
                raise ValueError("nse_symbol_registry is empty — using fallback lists.")

            log.info(
                f"Loaded {len(equities)} equities and {len(indices)} indices "
                "from nse_symbol_registry."
            )

        except Exception as e:
            log.warning(f"Could not load from nse_symbol_registry: {e}. Using fallback lists.")
            equities = _FALLBACK_EQUITIES
            indices  = _FALLBACK_INDICES

        return {"equities": equities, "indices": indices}

    # ─────────────────────────────────────────
    # Task 3: Fetch & store OHLCV prices (bulk)
    # ─────────────────────────────────────────

    @task()
    def ingest_prices(symbol_lists: dict, dag_run_id: str, symbol_type: str = "equities"):
        """
        Fetch 10-year OHLCV data for a symbol group (equities or indices).

        Strategy — per symbol:
          1. Fetch from yfinance.
          2. Save the DataFrame immediately into a staging table via pandas to_sql.
          3. Upsert from staging → price_history.
          4. Drop staging table.
          5. Sleep 30 seconds before the next symbol to avoid Yahoo rate limits.
        """
        import time
        from datetime import datetime
        import yfinance as yf
        import pandas as pd
        from sqlalchemy import text

        symbols = symbol_lists.get(symbol_type, [])
        if not symbols:
            log.warning(f"[price/{symbol_type}] No symbols provided — skipping.")
            return

        hook    = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        engine  = hook.get_sqlalchemy_engine()
        staging = f"_staging_price_{symbol_type}_tmp"

        success: list[str] = []
        failed:  list[str] = []

        for i, symbol in enumerate(symbols):
            # ── Check delta table: skip if already successfully ingested today ──
            already_done = hook.get_first("""
                SELECT data_to FROM ingestion_delta_table
                WHERE symbol    = %s
                  AND data_type = 'price'
                  AND status    = 'success'
                  AND end_time::date = CURRENT_DATE
            """, parameters=(symbol,))
            if already_done:
                log.info(
                    f"[price/{symbol_type}] ({i+1}/{len(symbols)}) "
                    f"Skipping {symbol} — already ingested today (data_to={already_done[0]})"
                )
                success.append(symbol)
                continue

            ingestion_start = datetime.now()
            try:
                df = yf.Ticker(symbol).history(period="10y", auto_adjust=False)

                if df.empty:
                    raise ValueError("Empty response from yfinance")

                df = (
                    df.reset_index()
                      .rename(columns={
                          "Date":      "date",
                          "Open":      "open",
                          "High":      "high",
                          "Low":       "low",
                          "Close":     "close",
                          "Adj Close": "adj_close",
                          "Volume":    "volume",
                      })
                )
                df["symbol"] = symbol
                df["date"]   = pd.to_datetime(df["date"]).dt.date
                df["volume"] = df["volume"].fillna(0).astype("int64")
                df = (
                    df[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]]
                    .dropna(subset=["open", "close"])
                )

                # ── Save this symbol's DataFrame to staging via to_sql ─────
                df.to_sql(
                    staging, engine,
                    if_exists="replace",
                    index=False,
                    method="multi",
                    chunksize=5000,
                )

                # ── Upsert from staging → price_history ───────────────────
                with engine.begin() as conn:
                    result = conn.execute(text(f"""
                        INSERT INTO price_history
                            (symbol, date, open, high, low, close, adj_close, volume)
                        SELECT symbol, date, open, high, low, close, adj_close, volume
                        FROM   {staging}
                        ON CONFLICT (symbol, date) DO UPDATE SET
                            open      = EXCLUDED.open,
                            high      = EXCLUDED.high,
                            low       = EXCLUDED.low,
                            close     = EXCLUDED.close,
                            adj_close = EXCLUDED.adj_close,
                            volume    = EXCLUDED.volume
                    """))
                    upserted = result.rowcount
                    conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))

                hook.run("""
                    INSERT INTO ingestion_delta_table
                        (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, dag_run_id)
                    VALUES (%s, 'price', %s, %s, %s, NOW(), 'success', %s, %s)
                """, parameters=(symbol, df["date"].min(), df["date"].max(), ingestion_start, upserted, dag_run_id))
                success.append(symbol)
                hook.insert_rows(
                    table="ingestion_log",
                    rows=[(symbol, "price", "success", upserted, None, dag_run_id)],
                    target_fields=["symbol", "data_type", "status", "rows_affected", "error_msg", "dag_run_id"],
                )
                log.info(
                    f"[price/{symbol_type}] ({i+1}/{len(symbols)}) "
                    f"{symbol}: {upserted} rows upserted"
                )

            except Exception as e:
                hook.run("""
                    INSERT INTO ingestion_delta_table
                        (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, error_msg, dag_run_id)
                    VALUES (%s, 'price', NULL, NULL, %s, NOW(), 'failed', 0, %s, %s)
                """, parameters=(symbol, ingestion_start, str(e)[:500], dag_run_id))
                failed.append(symbol)
                hook.insert_rows(
                    table="ingestion_log",
                    rows=[(symbol, "price", "failed", 0, str(e)[:200], dag_run_id)],
                    target_fields=["symbol", "data_type", "status", "rows_affected", "error_msg", "dag_run_id"],
                )
                log.error(f"[price/{symbol_type}] {symbol} failed: {e}")

            # ── Rate-limit guard: wait 30 s before the next request ────────
            if i < len(symbols) - 1:
                log.info(f"[price/{symbol_type}] Sleeping 30 s before next symbol...")
                time.sleep(30)

        log.info(
            f"[price/{symbol_type}] Done. "
            f"Success: {len(success)}, Failed: {len(failed)}"
        )

    # ─────────────────────────────────────────
    # Task 4: Fetch & store dividends (bulk)
    # ─────────────────────────────────────────

    @task()
    def ingest_dividends(symbol_lists: dict, dag_run_id: str):
        """
        Fetch full dividend history for all equities.

        Strategy — per symbol:
          1. Fetch dividends from yfinance.
          2. Save the DataFrame immediately via pandas to_sql into a staging table.
          3. Upsert from staging → dividends.
          4. Drop staging table.
          5. Sleep 30 seconds before the next symbol to avoid Yahoo rate limits.
        """
        import time
        from datetime import datetime
        import yfinance as yf
        import pandas as pd
        from sqlalchemy import text

        equities = symbol_lists.get("equities", [])
        if not equities:
            log.warning("[dividends] No equity symbols — skipping.")
            return

        hook    = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        engine  = hook.get_sqlalchemy_engine()
        staging = "_staging_dividends_tmp"

        success: list[str] = []
        failed:  list[str] = []

        for i, symbol in enumerate(equities):
            # ── Check delta table: skip if already successfully ingested today ──
            already_done = hook.get_first("""
                SELECT data_to FROM ingestion_delta_table
                WHERE symbol    = %s
                  AND data_type = 'dividend'
                  AND status    = 'success'
                  AND end_time::date = CURRENT_DATE
            """, parameters=(symbol,))
            if already_done:
                log.info(f"[dividends] ({i+1}/{len(equities)}) Skipping {symbol} — already ingested today")
                success.append(symbol)
                continue

            ingestion_start = datetime.now()
            try:
                divs = yf.Ticker(symbol).dividends
                if divs.empty:
                    hook.run("""
                        INSERT INTO ingestion_delta_table
                            (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, dag_run_id)
                        VALUES (%s, 'dividend', NULL, NULL, %s, NOW(), 'success', 0, %s)
                    """, parameters=(symbol, ingestion_start, dag_run_id))
                    log.info(f"[dividends] ({i+1}/{len(equities)}) No data for {symbol}")
                else:
                    df = divs.reset_index()
                    df.columns = ["ex_date", "amount"]
                    df["symbol"]  = symbol
                    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
                    df["amount"]  = df["amount"].astype(float)
                    df = df[["symbol", "ex_date", "amount"]]

                    # ── Save this symbol's DataFrame via to_sql ────────────
                    df.to_sql(
                        staging, engine,
                        if_exists="replace",
                        index=False,
                        method="multi",
                        chunksize=5000,
                    )

                    # ── Upsert from staging → dividends ───────────────────
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO dividends (symbol, ex_date, amount)
                            SELECT symbol, ex_date, amount
                            FROM   _staging_dividends_tmp
                            ON CONFLICT (symbol, ex_date) DO UPDATE SET
                                amount = EXCLUDED.amount
                        """))
                        conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))

                    hook.run("""
                        INSERT INTO ingestion_delta_table
                            (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, dag_run_id)
                        VALUES (%s, 'dividend', %s, %s, %s, NOW(), 'success', %s, %s)
                    """, parameters=(symbol, df["ex_date"].min(), df["ex_date"].max(), ingestion_start, len(df), dag_run_id))
                    success.append(symbol)
                    hook.insert_rows(
                        table="ingestion_log",
                        rows=[(symbol, "dividend", "success", len(df), None, dag_run_id)],
                        target_fields=["symbol", "data_type", "status", "rows_affected", "error_msg", "dag_run_id"],
                    )
                    log.info(f"[dividends] ({i+1}/{len(equities)}) {symbol}: {len(df)} rows upserted")

            except Exception as e:
                hook.run("""
                    INSERT INTO ingestion_delta_table
                        (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, error_msg, dag_run_id)
                    VALUES (%s, 'dividend', NULL, NULL, %s, NOW(), 'failed', 0, %s, %s)
                """, parameters=(symbol, ingestion_start, str(e)[:500], dag_run_id))
                failed.append(symbol)
                hook.insert_rows(
                    table="ingestion_log",
                    rows=[(symbol, "dividend", "failed", 0, str(e)[:200], dag_run_id)],
                    target_fields=["symbol", "data_type", "status", "rows_affected", "error_msg", "dag_run_id"],
                )
                log.error(f"[dividends] {symbol} failed: {e}")

            # ── Rate-limit guard ───────────────────────────────────────────
            if i < len(equities) - 1:
                log.info("[dividends] Sleeping 30 s before next symbol...")
                time.sleep(30)

        log.info(f"[dividends] Done. Success: {len(success)}, Failed: {len(failed)}")

    # ─────────────────────────────────────────
    # Task 5: Fetch & store splits (bulk)
    # ─────────────────────────────────────────

    @task()
    def ingest_splits(symbol_lists: dict, dag_run_id: str):
        """
        Fetch full stock split history for all equities.

        Strategy — per symbol:
          1. Fetch splits from yfinance.
          2. Save the DataFrame immediately via pandas to_sql into a staging table.
          3. Upsert from staging → splits.
          4. Drop staging table.
          5. Sleep 30 seconds before the next symbol to avoid Yahoo rate limits.
        """
        import time
        from datetime import datetime
        import yfinance as yf
        import pandas as pd
        from sqlalchemy import text

        equities = symbol_lists.get("equities", [])
        if not equities:
            log.warning("[splits] No equity symbols — skipping.")
            return

        hook    = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        engine  = hook.get_sqlalchemy_engine()
        staging = "_staging_splits_tmp"

        success: list[str] = []
        failed:  list[str] = []

        for i, symbol in enumerate(equities):
            # ── Check delta table: skip if already successfully ingested today ──
            already_done = hook.get_first("""
                SELECT data_to FROM ingestion_delta_table
                WHERE symbol    = %s
                  AND data_type = 'split'
                  AND status    = 'success'
                  AND end_time::date = CURRENT_DATE
            """, parameters=(symbol,))
            if already_done:
                log.info(f"[splits] ({i+1}/{len(equities)}) Skipping {symbol} — already ingested today")
                success.append(symbol)
                continue

            ingestion_start = datetime.now()
            try:
                splits = yf.Ticker(symbol).splits
                if splits.empty:
                    hook.run("""
                        INSERT INTO ingestion_delta_table
                            (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, dag_run_id)
                        VALUES (%s, 'split', NULL, NULL, %s, NOW(), 'success', 0, %s)
                    """, parameters=(symbol, ingestion_start, dag_run_id))
                    log.info(f"[splits] ({i+1}/{len(equities)}) No data for {symbol}")
                else:
                    df = splits.reset_index()
                    df.columns = ["split_date", "ratio"]
                    df["symbol"]     = symbol
                    df["split_date"] = pd.to_datetime(df["split_date"]).dt.date
                    df["ratio"]      = df["ratio"].astype(float)
                    df = df[["symbol", "split_date", "ratio"]]

                    # ── Save this symbol's DataFrame via to_sql ────────────
                    df.to_sql(
                        staging, engine,
                        if_exists="replace",
                        index=False,
                        method="multi",
                        chunksize=5000,
                    )

                    # ── Upsert from staging → splits ───────────────────────
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO splits (symbol, split_date, ratio)
                            SELECT symbol, split_date, ratio
                            FROM   _staging_splits_tmp
                            ON CONFLICT (symbol, split_date) DO UPDATE SET
                                ratio = EXCLUDED.ratio
                        """))
                        conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))

                    hook.run("""
                        INSERT INTO ingestion_delta_table
                            (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, dag_run_id)
                        VALUES (%s, 'split', %s, %s, %s, NOW(), 'success', %s, %s)
                    """, parameters=(symbol, df["split_date"].min(), df["split_date"].max(), ingestion_start, len(df), dag_run_id))
                    success.append(symbol)
                    hook.insert_rows(
                        table="ingestion_log",
                        rows=[(symbol, "split", "success", len(df), None, dag_run_id)],
                        target_fields=["symbol", "data_type", "status", "rows_affected", "error_msg", "dag_run_id"],
                    )
                    log.info(f"[splits] ({i+1}/{len(equities)}) {symbol}: {len(df)} rows upserted")

            except Exception as e:
                hook.run("""
                    INSERT INTO ingestion_delta_table
                        (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, error_msg, dag_run_id)
                    VALUES (%s, 'split', NULL, NULL, %s, NOW(), 'failed', 0, %s, %s)
                """, parameters=(symbol, ingestion_start, str(e)[:500], dag_run_id))
                failed.append(symbol)
                hook.insert_rows(
                    table="ingestion_log",
                    rows=[(symbol, "split", "failed", 0, str(e)[:200], dag_run_id)],
                    target_fields=["symbol", "data_type", "status", "rows_affected", "error_msg", "dag_run_id"],
                )
                log.error(f"[splits] {symbol} failed: {e}")

            # ── Rate-limit guard ───────────────────────────────────────────
            if i < len(equities) - 1:
                log.info("[splits] Sleeping 30 s before next symbol...")
                time.sleep(30)

        log.info(f"[splits] Done. Success: {len(success)}, Failed: {len(failed)}")

    # ─────────────────────────────────────────
    # Task 6: Fetch & store fundamentals (bulk)
    # ─────────────────────────────────────────

    @task()
    def ingest_fundamentals(symbol_lists: dict, dag_run_id: str):
        """
        Fetch current fundamentals from ticker.info for all equities.
        Accumulates all rows then bulk-upserts via hook.insert_rows.
        Sleeps 30 seconds between symbols to avoid Yahoo rate limits.
        """
        import time
        from datetime import date, datetime
        import yfinance as yf

        equities = symbol_lists.get("equities", [])
        if not equities:
            log.warning("[fundamentals] No equity symbols — skipping.")
            return

        hook  = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        today = date.today()

        for i, symbol in enumerate(equities):
            # ── Check delta table: skip if already successfully ingested today ──
            already_done = hook.get_first("""
                SELECT data_to FROM ingestion_delta_table
                WHERE symbol    = %s
                  AND data_type = 'fundamental'
                  AND status    = 'success'
                  AND end_time::date = CURRENT_DATE
            """, parameters=(symbol,))
            if already_done:
                log.info(f"[fundamentals] ({i+1}/{len(equities)}) Skipping {symbol} — already ingested today")
                continue

            ingestion_start = datetime.now()
            try:
                info = yf.Ticker(symbol).info
                hook.insert_rows(
                    table="fundamentals",
                    rows=[(
                        symbol, today,
                        info.get("trailingPE"),
                        info.get("priceToBook"),
                        info.get("trailingEps"),
                        info.get("totalRevenue"),
                        info.get("netIncomeToCommon"),
                        info.get("debtToEquity"),
                        info.get("returnOnEquity"),
                        info.get("marketCap"),
                    )],
                    target_fields=[
                        "symbol", "as_of_date", "pe_ratio", "pb_ratio", "eps",
                        "revenue", "net_income", "debt_to_equity", "roe", "market_cap",
                    ],
                    replace=True,
                    replace_index=["symbol", "as_of_date"],
                )
                hook.run("""
                    INSERT INTO ingestion_delta_table
                        (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, dag_run_id)
                    VALUES (%s, 'fundamental', %s, %s, %s, NOW(), 'success', 1, %s)
                """, parameters=(symbol, today, today, ingestion_start, dag_run_id))
                hook.insert_rows(
                    table="ingestion_log",
                    rows=[(symbol, "fundamental", "success", 1, None, dag_run_id)],
                    target_fields=["symbol", "data_type", "status", "rows_affected", "error_msg", "dag_run_id"],
                )
                log.info(f"[fundamentals] ({i+1}/{len(equities)}) Fetched: {symbol}")

            except Exception as e:
                hook.run("""
                    INSERT INTO ingestion_delta_table
                        (symbol, data_type, data_from, data_to, start_time, end_time, status, record_count, error_msg, dag_run_id)
                    VALUES (%s, 'fundamental', NULL, NULL, %s, NOW(), 'failed', 0, %s, %s)
                """, parameters=(symbol, ingestion_start, str(e)[:500], dag_run_id))
                hook.insert_rows(
                    table="ingestion_log",
                    rows=[(symbol, "fundamental", "failed", 0, str(e)[:200], dag_run_id)],
                    target_fields=["symbol", "data_type", "status", "rows_affected", "error_msg", "dag_run_id"],
                )
                log.error(f"[fundamentals] {symbol} failed: {e}")

            # ── Rate-limit guard ───────────────────────────────────────────
            if i < len(equities) - 1:
                log.info("[fundamentals] Sleeping 30 s before next symbol...")
                time.sleep(30)


    # ─────────────────────────────────────────
    # Task 7: Summary report
    # ─────────────────────────────────────────

    @task()
    def ingestion_summary(dag_run_id: str):
        """
        Query ingestion_log and print a summary of this run.
        Useful for monitoring and alerting.
        """
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        rows = hook.get_records("""
            SELECT data_type, status, COUNT(*) as count, SUM(rows_affected) as total_rows
            FROM ingestion_log
            WHERE dag_run_id = %s
            GROUP BY data_type, status
            ORDER BY data_type, status
        """, parameters=(dag_run_id,))

        log.info("=" * 50)
        log.info(f"INGESTION SUMMARY — Run ID: {dag_run_id}")
        log.info("=" * 50)
        for data_type, status, count, total_rows in rows:
            log.info(f"  {data_type:<15} | {status:<8} | symbols: {count:<4} | rows: {total_rows}")
        log.info("=" * 50)

    # ─────────────────────────────────────────
    # Task flow / dependencies
    # ─────────────────────────────────────────

    dag_run_id = "{{ run_id }}"

    # Step 1: ensure tables exist
    tables_ready = validate_db_tables()

    # Step 2: fetch symbol lists from nse_symbol_registry
    symbol_lists = fetch_symbols_from_db(tables_ready)

    # Step 3a: ingest 1-year prices for equities
    equity_prices = ingest_prices.override(task_id="ingest_equity_prices")(
        symbol_lists=symbol_lists,
        dag_run_id=dag_run_id,
        symbol_type="equities",
    )

    # Step 3b: ingest 1-year prices for indices (parallel with equities)
    index_prices = ingest_prices.override(task_id="ingest_index_prices")(
        symbol_lists=symbol_lists,
        dag_run_id=dag_run_id,
        symbol_type="indices",
    )

    # Step 4 & 5: dividends and splits (parallel)
    dividends = ingest_dividends(symbol_lists=symbol_lists, dag_run_id=dag_run_id)
    splits    = ingest_splits(symbol_lists=symbol_lists, dag_run_id=dag_run_id)

    # Step 6: fundamentals (parallel)
    fundamentals = ingest_fundamentals(symbol_lists=symbol_lists, dag_run_id=dag_run_id)

    # Step 7: summary after everything completes
    summary = ingestion_summary(dag_run_id=dag_run_id)

    [equity_prices, index_prices, dividends, splits, fundamentals] >> summary


# Register the DAG
indian_stock_data_ingestion()
