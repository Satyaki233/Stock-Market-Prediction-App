"""
NSE Symbol Registry DAG
Airflow 3.x (uses airflow.sdk imports)

Purpose:
    Single source of truth for ALL NSE symbols.
    Downstream DAGs (price ingestion, etc.) query `nse_symbol_registry`
    instead of using hardcoded lists.

Covers:
    - All NSE-listed equities (downloaded from NSE official CSV)
    - NSE broad market indices
    - NSE sector indices
    - NSE thematic / strategy indices
    - Currency & volatility symbols

Schedule: Weekly on Sunday at 00:00 UTC
    (Symbol list changes rarely; weekly refresh is sufficient)
"""

import logging
from datetime import datetime

import pendulum
from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

log = logging.getLogger(__name__)

POSTGRES_CONN_ID = "stock_db_conn"

# ─────────────────────────────────────────────
# Comprehensive NSE Indices
# Format: (yfinance_ticker, display_name, index_category)
# ─────────────────────────────────────────────

NSE_BROAD_INDICES = [
    ("^NSEI",       "Nifty 50",                 "broad"),
    ("^NSEMDCP50",  "Nifty Midcap 50",          "broad"),
    ("^NSMIDCP",    "Nifty Midcap 100",         "broad"),
    ("^CNXSC",      "Nifty Smallcap 100",       "broad"),
    ("^CNX100",     "Nifty 100",                "broad"),
    ("^CNX200",     "Nifty 200",                "broad"),
    ("^CNX500",     "Nifty 500",                "broad"),
    ("^CNXMICRO",   "Nifty Microcap 250",       "broad"),
    ("^BSESN",      "BSE Sensex",               "broad"),
    ("^INDIAVIX",   "India VIX",                "volatility"),
    ("INR=X",       "USD/INR",                  "currency"),
]

NSE_SECTOR_INDICES = [
    ("^NSEBANK",       "Nifty Bank",                    "sector"),
    ("^CNXIT",         "Nifty IT",                      "sector"),
    ("^CNXPHARMA",     "Nifty Pharma",                  "sector"),
    ("^CNXAUTO",       "Nifty Auto",                    "sector"),
    ("^CNXFMCG",       "Nifty FMCG",                    "sector"),
    ("^CNXMETAL",      "Nifty Metal",                   "sector"),
    ("^CNXENERGY",     "Nifty Energy",                  "sector"),
    ("^CNXREALTY",     "Nifty Realty",                  "sector"),
    ("^CNXMEDIA",      "Nifty Media",                   "sector"),
    ("^CNXINFRA",      "Nifty Infrastructure",          "sector"),
    ("^CNXPSUBANK",    "Nifty PSU Bank",                "sector"),
    ("^CNXFINANCE",    "Nifty Financial Services",      "sector"),
    ("^CNXCONSUMPTION","Nifty India Consumption",       "sector"),
    ("^CNXCOMMODITIES","Nifty Commodities",             "sector"),
    ("^CNXMNCAP",      "Nifty MNC",                     "sector"),
    ("^CNXDIVOPPT",    "Nifty Dividend Opportunities",  "sector"),
    ("^CNXSERVICE",    "Nifty Services Sector",         "sector"),
    ("^CNXPSE",        "Nifty PSE",                     "sector"),
]

NSE_THEMATIC_INDICES = [
    ("^CNXPVTBANK",    "Nifty Private Bank",            "thematic"),
    ("^CNXFINSRV25",   "Nifty Financial Services 25/50","thematic"),
    ("^CNXHEALTHCARE", "Nifty Healthcare",              "thematic"),
    ("^CNXOILGAS",     "Nifty Oil & Gas",               "thematic"),
    ("^CNXCHEMS",      "Nifty Chemicals",               "thematic"),
    ("^NIFTYINDIA",    "Nifty India Digital",           "thematic"),
    ("^CNXSMALLCAP250","Nifty Smallcap 250",            "thematic"),
    ("^CNXMIDSMALL400","Nifty Midsmallcap 400",         "thematic"),
    ("^CNXTOTAL",      "Nifty Total Market",            "thematic"),
    ("^CNXALPHA",      "Nifty Alpha 50",                "thematic"),
    ("^CNXHIGHBETA",   "Nifty High Beta 50",            "thematic"),
    ("^CNXLOWVOL",     "Nifty Low Volatility 50",       "thematic"),
    ("^CNXQUAL30",     "Nifty Quality 30",              "thematic"),
    ("^CNXVALUE20",    "Nifty Value 20",                "thematic"),
    ("^CNXGROWTH",     "Nifty Growth Sectors 15",       "thematic"),
    ("^CNXMIDCAP150",  "Nifty Midcap 150",              "thematic"),
    ("^CNXSMALLCAP50", "Nifty Smallcap 50",             "thematic"),
    ("^CNXLCAP",       "Nifty Largecap 250",            "thematic"),
]

ALL_INDICES = NSE_BROAD_INDICES + NSE_SECTOR_INDICES + NSE_THEMATIC_INDICES


# ─────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────

@dag(
    dag_id="nse_symbol_registry",
    description="Build and maintain a complete NSE symbol registry (equities + all indices)",
    schedule="0 0 * * 0",              # Every Sunday at midnight UTC
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["stocks", "registry", "nse", "india"],
    default_args={
        "retries": 2,
        "retry_delay": pendulum.duration(minutes=10),
        "owner": "data-team",
    },
)
def nse_symbol_registry():

    # ──────────────────────────────────────────
    # Task 1: Create registry table
    # ──────────────────────────────────────────

    @task()
    def create_registry_table():
        """
        Create nse_symbol_registry if it doesn't exist.
        This is the master symbol table all other DAGs query.
        """
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        hook.run("""
            CREATE TABLE IF NOT EXISTS nse_symbol_registry (
                id              SERIAL,
                symbol          VARCHAR(30)  NOT NULL,
                yf_symbol       VARCHAR(30)  NOT NULL,   -- ticker used with yfinance
                name            VARCHAR(300),
                symbol_type     VARCHAR(20)  NOT NULL,   -- 'equity', 'index', 'etf'
                index_category  VARCHAR(20),             -- 'broad','sector','thematic','volatility','currency'
                sector          VARCHAR(100),
                industry        VARCHAR(200),
                isin            VARCHAR(20),
                series          VARCHAR(5),              -- EQ, BE, etc.
                listing_date    DATE,
                face_value      NUMERIC(10,2),
                paid_up_value   NUMERIC(10,2),
                market_lot      INTEGER,
                is_active       BOOLEAN      DEFAULT TRUE,
                last_updated    TIMESTAMP    DEFAULT NOW(),
                highlight       BOOLEAN      DEFAULT FALSE,
                PRIMARY KEY (symbol, symbol_type)
            );

            CREATE INDEX IF NOT EXISTS idx_registry_type
                ON nse_symbol_registry(symbol_type);
            CREATE INDEX IF NOT EXISTS idx_registry_sector
                ON nse_symbol_registry(sector);
            CREATE INDEX IF NOT EXISTS idx_registry_yf
                ON nse_symbol_registry(yf_symbol);
        """)
        log.info("nse_symbol_registry table ready.")
        return "table_ready"

    # ──────────────────────────────────────────
    # Task 2: Fetch all NSE equities from NSE CSV
    # ──────────────────────────────────────────

    @task()
    def fetch_nse_equities(_: str):
        """
        Download the complete NSE equity list from NSE's official CSV endpoint.
        URL: https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
        CSV columns: SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING,
                     PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE

        Inserts every row as symbol_type='equity' with yf_symbol = SYMBOL + '.NS'
        """
        import io
        import requests
        import pandas as pd

        NSE_EQUITY_CSV_URL = (
            "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        )

        # NSE requires browser-like headers
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer":         "https://www.nseindia.com/",
            "Connection":      "keep-alive",
        }

        # First hit the NSE homepage to get cookies, then fetch the CSV
        session = requests.Session()
        try:
            session.get("https://www.nseindia.com", headers=headers, timeout=15)
        except Exception as e:
            log.warning(f"NSE homepage prefetch failed (non-fatal): {e}")

        resp = session.get(NSE_EQUITY_CSV_URL, headers=headers, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(
            io.StringIO(resp.text),
            dtype=str,
        )

        # Normalise column names: strip whitespace
        df.columns = [c.strip() for c in df.columns]

        log.info(f"NSE equity CSV fetched: {len(df)} rows, columns: {list(df.columns)}")

        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        upserted = 0
        skipped  = 0

        for _, row in df.iterrows():
            symbol = str(row.get("SYMBOL", "")).strip()
            if not symbol:
                skipped += 1
                continue

            name         = str(row.get("NAME OF COMPANY",  "")).strip() or None
            series       = str(row.get("SERIES",           "")).strip() or None
            isin         = str(row.get("ISIN NUMBER",       "")).strip() or None
            yf_symbol    = f"{symbol}.NS"

            # Numeric fields — coerce safely
            def _num(col):
                try:
                    return float(str(row.get(col, "")).strip())
                except (ValueError, TypeError):
                    return None

            def _date(col):
                raw = str(row.get(col, "")).strip()
                if not raw or raw.lower() == "nan":
                    return None
                try:
                    return pd.to_datetime(raw, dayfirst=True).date()
                except Exception:
                    return None

            listing_date  = _date("DATE OF LISTING")
            face_value    = _num("FACE VALUE")
            paid_up_value = _num("PAID UP VALUE")
            market_lot    = int(row.get("MARKET LOT", 0) or 0) or None

            hook.run("""
                INSERT INTO nse_symbol_registry
                    (symbol, yf_symbol, name, symbol_type,
                     isin, series, listing_date,
                     face_value, paid_up_value, market_lot,
                     is_active, last_updated)
                VALUES (%s, %s, %s, 'equity',
                        %s, %s, %s,
                        %s, %s, %s,
                        TRUE, NOW())
                ON CONFLICT (symbol, symbol_type) DO UPDATE SET
                    yf_symbol     = EXCLUDED.yf_symbol,
                    name          = EXCLUDED.name,
                    isin          = EXCLUDED.isin,
                    series        = EXCLUDED.series,
                    listing_date  = EXCLUDED.listing_date,
                    face_value    = EXCLUDED.face_value,
                    paid_up_value = EXCLUDED.paid_up_value,
                    market_lot    = EXCLUDED.market_lot,
                    is_active     = TRUE,
                    last_updated  = NOW()
            """, parameters=(
                symbol, yf_symbol, name,
                isin, series, listing_date,
                face_value, paid_up_value, market_lot,
            ))
            upserted += 1

        log.info(f"[equities] Upserted: {upserted}, Skipped: {skipped}")
        return upserted

    # ──────────────────────────────────────────
    # Task 3: Upsert all NSE indices
    # ──────────────────────────────────────────

    @task()
    def upsert_nse_indices(_: str):
        """
        Insert / update the comprehensive NSE index list.
        yf_symbol IS the index ticker (e.g. ^NSEI, ^CNXIT).
        symbol_type = 'index'
        """
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        upserted = 0

        for yf_ticker, display_name, category in ALL_INDICES:
            # Use the yf_ticker stripped of ^ as the canonical symbol key
            symbol = yf_ticker.lstrip("^").replace("=", "_")

            hook.run("""
                INSERT INTO nse_symbol_registry
                    (symbol, yf_symbol, name, symbol_type, index_category,
                     is_active, last_updated)
                VALUES (%s, %s, %s, 'index', %s, TRUE, NOW())
                ON CONFLICT (symbol, symbol_type) DO UPDATE SET
                    yf_symbol      = EXCLUDED.yf_symbol,
                    name           = EXCLUDED.name,
                    index_category = EXCLUDED.index_category,
                    is_active      = TRUE,
                    last_updated   = NOW()
            """, parameters=(symbol, yf_ticker, display_name, category))
            upserted += 1

        log.info(f"[indices] Upserted: {upserted} index symbols")
        return upserted

    # ──────────────────────────────────────────
    # Task 4: Enrich equities with sector/industry from yfinance
    # ──────────────────────────────────────────

    @task()
    def enrich_equity_metadata(equity_count: int):
        """
        For equities that are missing sector/industry data, fetch from yfinance.
        To avoid rate-limiting, processes only symbols missing sector info.
        Caps at 500 per run — subsequent runs fill in the rest.
        """
        import yfinance as yf
        import time

        if equity_count == 0:
            log.info("No equities to enrich.")
            return

        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        # Fetch symbols that need enrichment
        rows = hook.get_records("""
            SELECT symbol, yf_symbol
            FROM nse_symbol_registry
            WHERE symbol_type = 'equity'
              AND (sector IS NULL OR industry IS NULL)
              AND series = 'EQ'          -- only main board equities
            ORDER BY listing_date DESC NULLS LAST
            LIMIT 500
        """)

        log.info(f"[enrich] {len(rows)} equities need sector/industry enrichment")
        enriched = 0
        failed   = 0

        for symbol, yf_symbol in rows:
            try:
                info     = yf.Ticker(yf_symbol).info
                sector   = info.get("sector")   or None
                industry = info.get("industry") or None
                name     = info.get("longName")  or None

                hook.run("""
                    UPDATE nse_symbol_registry
                    SET sector   = COALESCE(%s, sector),
                        industry = COALESCE(%s, industry),
                        name     = COALESCE(%s, name),
                        last_updated = NOW()
                    WHERE symbol = %s AND symbol_type = 'equity'
                """, parameters=(sector, industry, name, symbol))

                enriched += 1
                # Polite rate-limit: 0.3 s per ticker
                time.sleep(0.3)

            except Exception as e:
                failed += 1
                log.warning(f"[enrich] {symbol} failed: {e}")

        log.info(f"[enrich] Done. Enriched: {enriched}, Failed: {failed}")

    # ──────────────────────────────────────────
    # Task 5: Registry summary
    # ──────────────────────────────────────────

    @task()
    def registry_summary():
        """Print a count breakdown of the registry by type and category."""
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        rows = hook.get_records("""
            SELECT
                symbol_type,
                index_category,
                COUNT(*) AS total,
                COUNT(sector) AS with_sector
            FROM nse_symbol_registry
            WHERE is_active = TRUE
            GROUP BY symbol_type, index_category
            ORDER BY symbol_type, index_category
        """)

        log.info("=" * 60)
        log.info("NSE SYMBOL REGISTRY — SUMMARY")
        log.info("=" * 60)
        for sym_type, cat, total, with_sector in rows:
            cat_str = f"[{cat}]" if cat else ""
            log.info(f"  {sym_type:<10} {cat_str:<15} total: {total:<6} with_sector: {with_sector}")
        log.info("=" * 60)

    # ──────────────────────────────────────────
    # Task flow
    # ──────────────────────────────────────────

    table_ready    = create_registry_table()
    equity_count   = fetch_nse_equities(table_ready)
    index_count    = upsert_nse_indices(table_ready)   # parallel with equities
    enriched       = enrich_equity_metadata(equity_count)
    summary        = registry_summary()

    # index upsert and enrichment both feed into summary
    [enriched, index_count] >> summary


# Register the DAG
nse_symbol_registry()
