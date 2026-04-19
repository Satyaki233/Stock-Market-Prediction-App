# Indian Stock Market Prediction App

An end-to-end ML pipeline that ingests Indian stock market data, engineers features, trains a direction-prediction model, and serves predictions — fully orchestrated with Apache Airflow.

---

## Technologies Used

| | Technology | Version | Role |
|---|---|---|---|
| **Orchestration** | Apache Airflow | 3.1.7 | Schedules and monitors the entire pipeline via DAGs |
| **Task Queue** | Redis | 7.2 | Celery message broker for distributing Airflow tasks across workers |
| **Data Source** | yfinance | 1.3.0 | Fetches OHLCV, dividends, splits, and fundamentals from Yahoo Finance |
| **Primary Database** | PostgreSQL | 16 | Stores raw market data, engineered features, and predictions |
| **ML Framework** | LightGBM | 4.5.0 | Gradient boosting classifier for next-day direction prediction |
| **ML Ops** | MLflow | 2.19.0 | Tracks experiments, logs metrics/artifacts, and registers models |
| **Serving Database** | MongoDB | 7 | Document store for serving predictions to the API layer |
| **Data Processing** | pandas + NumPy | latest | Feature computation and data wrangling |
| **Containerisation** | Docker + Compose | latest | Runs every service in isolated, reproducible containers |
| **Python** | Python | 3.12 | Runtime for all DAGs and ML code |

---

## What We Are Building

We predict whether a given NSE-listed stock will close **up or down the next trading day**. The entire pipeline runs automatically every weekday, keeping predictions fresh without any manual intervention.

---

## Architecture Overview

```
Yahoo Finance API (yfinance)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Apache Airflow                           │
│                                                             │
│  DAG 1: NSE Symbol Registry  (weekly)                       │
│       → populates nse_symbol_registry table                 │
│                                                             │
│  DAG 2: Data Ingestion  (daily, 4:30 PM IST)                │
│       → price_history, dividends, splits, fundamentals      │
│                                                             │
│  DAG 3: Feature Engineering  (daily, 5:00 PM IST)           │
│       → stock_features  (50+ ML-ready columns)              │
│                                                             │
│  DAG 4: Model Training  (daily, 6:00 PM IST)                │
│       → LightGBM classifier + MLflow experiment tracking    │
│       → stock_predictions                                   │
│                                                             │
│  DAG 5: MongoDB Sync  (daily, 6:30 PM IST)                  │
│       → pushes predictions to MongoDB for API consumption   │
└─────────────────────────────────────────────────────────────┘
        │                          │
        ▼                          ▼
  PostgreSQL (stocks_db)      MongoDB (stock_db)
  raw data + features         predictions collection
        │
        ▼
  MLflow (experiment tracker)
  model registry + artifacts
```

---

## Pipeline — Step by Step

### Step 1 — NSE Symbol Registry
**DAG:** `nse_symbol_registry` · **Schedule:** Weekly (Sunday 00:00 UTC)

Builds and maintains a single source of truth for all tradeable NSE symbols — equities, broad-market indices, sector indices, and thematic indices. Downstream DAGs query this table instead of using hardcoded symbol lists.

---

### Step 2 — Data Ingestion
**DAG:** `indian_stock_data_ingestion` · **Schedule:** Daily at 11:00 UTC (4:30 PM IST)

Pulls 10 years of historical data from the **Yahoo Finance API (yfinance)** for every active symbol in the registry. Stores four data types in PostgreSQL:

| Table | Contents |
|---|---|
| `price_history` | OHLCV (Open, High, Low, Close, Volume) — daily bars |
| `dividends` | Historical dividend events per symbol |
| `splits` | Stock split history per symbol |
| `fundamentals` | PE ratio, PB ratio, EPS, Revenue, Debt-to-Equity, ROE, Market Cap |

A delta table (`ingestion_delta_table`) tracks what was already ingested so re-runs skip symbols that were successfully processed today.

---

### Step 3 — Feature Engineering
**DAG:** `stock_feature_engineering` · **Schedule:** Daily at 11:30 UTC (5:00 PM IST)

Reads raw tables and computes 50+ ML-ready features into the `stock_features` table:

| Category | Features |
|---|---|
| Returns | `daily_return`, `log_return`, `weekly_return`, `monthly_return` |
| Moving Averages | `sma_5/10/20/50/200`, `ema_12/26` |
| Momentum | `macd`, `macd_signal`, `macd_hist`, `rsi_14` |
| Volatility | `bb_upper/lower/width/pct`, `atr_14`, `rolling_std_5/20` |
| Oscillators | `stoch_k`, `stoch_d` |
| Volume | `volume_sma_20`, `volume_ratio` |
| Price Levels | `high_52w`, `low_52w`, `pct_from_52w_high`, `pct_from_52w_low` |
| Lag Features | `close_lag_1/2/5`, `return_lag_1/2/5` |
| Fundamentals | `pe_ratio`, `pb_ratio`, `eps`, `revenue`, `net_income`, `debt_to_equity`, `roe`, `market_cap` |
| Events | `dividend_yield_ttm`, `days_since_dividend`, `days_since_split` |
| Metadata | `sector`, `industry` (categorical) |
| **Targets** | `target_direction_1d` (1=up / 0=down), `target_direction_5d`, `target_return_1d/5d` |

---

### Step 4 — Model Training
**DAG:** `stock_model_training` · **Schedule:** Daily at 12:30 UTC (6:00 PM IST)

Trains a **LightGBM binary classifier** to predict `target_direction_1d` — whether each stock closes higher the next trading day.

**Training approach:**
- **Global model** — trained on all symbols simultaneously, not one model per stock
- **3-fold walk-forward cross-validation** — chronological splits, zero data leakage
- **Final model** — retrained on 90% of history, validated on the last 10%

**MLflow tracks every run:**
- Hyperparameters
- Per-fold and average CV metrics (accuracy, AUC-ROC, F1, log-loss)
- Final validation metrics
- Feature importance table (CSV artifact)
- Trained model artifact (registered in MLflow Model Registry)

Predictions are written to `stock_predictions` in PostgreSQL:

| Column | Description |
|---|---|
| `symbol` | NSE ticker |
| `feature_date` | Date of the features used (last known trading day) |
| `prediction_date` | Next business day being predicted |
| `predicted_direction` | `1` = UP, `0` = DOWN |
| `prob_up` | Model confidence that the stock goes up |
| `prob_down` | Model confidence that the stock goes down |

---

### Step 5 — MongoDB Sync
**DAG:** `predictions_to_mongodb` · **Schedule:** Daily at 13:00 UTC (6:30 PM IST)

Reads all rows from `stock_predictions` (PostgreSQL) and bulk-upserts them into **MongoDB `stock_db.predictions`** collection. Uses `(symbol, feature_date)` as the unique key so re-runs are fully idempotent.

MongoDB is used as the serving layer because it is schemaless, horizontally scalable, and fast for document lookups by symbol — ideal for an API backend.

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | Apache Airflow 3.1.7 (CeleryExecutor) |
| Data Source | Yahoo Finance via `yfinance` |
| Primary Database | PostgreSQL 16 |
| ML Framework | LightGBM |
| Experiment Tracking | MLflow 2.19.0 |
| Serving Database | MongoDB 7 |
| Containerisation | Docker + Docker Compose |
| Task Queue | Redis (Celery broker) |

---

## Repository Structure

```
Stock-Market-Prediction-App/
│
├── Airflow/                        # Airflow project
│   ├── Dockerfile                  # Custom image with ML dependencies
│   ├── docker-compose.yaml         # Airflow + PostgreSQL + Redis + MLflow
│   ├── requirements.txt            # Pinned Python dependencies
│   │
│   ├── dags/
│   │   ├── nse_symbol_registry_dag.py      # Step 1 — symbol universe
│   │   ├── data_ingestion_dag.py           # Step 2 — raw data from yfinance
│   │   ├── feature_engineering_dag.py      # Step 3 — ML feature table
│   │   ├── model_training_dag.py           # Step 4 — LightGBM + MLflow
│   │   ├── predictions_to_mongodb_dag.py   # Step 5 — push to MongoDB
│   │   └── utils/
│   │       └── db_manager.py               # PostgreSQL connection helper
│   │
│   └── docker-entrypoint-initdb.d/
│       └── create_stocks_db.sql            # Creates stocks_db on first boot
│
└── Databases/                      # Standalone database containers
    ├── docker-compose.yaml         # PostgreSQL (stock_db, port 5433)
    └── docker-compose.mongo.yaml   # MongoDB (stock_db, port 27017)
```

---

## Running the Stack

### 1. Start databases
```bash
cd Databases
docker compose up -d                        # PostgreSQL on :5433
docker compose -f docker-compose.mongo.yaml up -d   # MongoDB on :27017
```

### 2. Build and start Airflow
```bash
cd Airflow
docker build .
docker compose up -d
```

### 3. Open the Airflow UI
```
http://localhost:8080
Username: airflow  |  Password: airflow
```

### 4. Add the PostgreSQL connection (once)
Admin → Connections → Add:

| Field | Value |
|---|---|
| Conn ID | `stock_db_conn` |
| Conn Type | `Postgres` |
| Host | `postgres` |
| Port | `5432` |
| Login | `airflow` |
| Password | `airflow` |
| Schema | `stocks_db` |

### 5. Add the MongoDB connection (once)
Admin → Connections → Add:

| Field | Value |
|---|---|
| Conn ID | `backend_app_db_conn` |
| Conn Type | `MongoDB` |
| Host | `host.docker.internal` |
| Port | `27017` |
| Login | `admin` |
| Password | `admin` |
| Schema | `stock_db` |

### 6. Enable and trigger DAGs in order
1. `nse_symbol_registry`
2. `indian_stock_data_ingestion`
3. `stock_feature_engineering`
4. `stock_model_training`
5. `predictions_to_mongodb`

### 7. Monitor experiments
```
http://localhost:5000   (MLflow UI)
```

---

## Future Scope

### FastAPI Backend
A REST API layer will sit in front of MongoDB and expose prediction data to clients (web apps, mobile, dashboards). Planned endpoints:

```
GET /predictions/latest          — latest prediction for all symbols
GET /predictions/{symbol}        — full history for a single symbol
GET /predictions/date/{date}     — all predictions for a given trading date
```

### Caching Layer
To reduce MongoDB load and avoid redundant reads on high-traffic endpoints, a **Redis cache** will be added between FastAPI and MongoDB:

```
Client → FastAPI → Redis (cache hit?) → MongoDB (cache miss)
                       ↑
               TTL-based invalidation
               refreshed after each daily DAG run
```

This is well-suited here because predictions are written once per day — cache entries can have a 24-hour TTL, making read traffic almost entirely served from memory.
