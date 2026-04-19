"""
Stock Predictions → MongoDB Sync DAG
Airflow 3.x (uses airflow.sdk imports)

Schedule : Daily at 1:00 PM UTC (6:30 PM IST) — 30 min after model training (12:30 UTC)
Source   : stock_predictions (PostgreSQL, conn: stock_db_conn)
Target   : MongoDB stock_db.predictions (conn: mongo_stock_db)

Task flow
─────────
  fetch_predictions
        ↓
  push_to_mongodb

Connection setup (do once in Airflow UI → Admin → Connections):
  Conn ID   : mongo_stock_db
  Conn Type : MongoDB
  Host      : host.docker.internal   ← reaches the Mac host from inside Docker
  Port      : 27017
  Login     : admin
  Password  : admin
  Schema    : stock_db               ← default database

NOTE: MongoDB runs in a separate Docker Compose project (Databases/) on a
different network. Airflow containers cannot reach it by service name, so
host.docker.internal is used to route through the Mac host machine.
"""

import logging

import pendulum
from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

log = logging.getLogger(__name__)

POSTGRES_CONN_ID = "stock_db_conn"
MONGO_CONN_ID    = "backend_app_db_conn"
MONGO_DB         = "stock_db"
MONGO_COLLECTION = "predictions"


@dag(
    dag_id="predictions_to_mongodb",
    description="Sync stock_predictions from PostgreSQL into MongoDB stock_db.predictions",
    schedule="0 13 * * 1-5",           # 1:00 PM UTC = 6:30 PM IST, weekdays
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["stocks", "mongodb", "predictions", "sync"],
    default_args={
        "retries": 2,
        "retry_delay": pendulum.duration(minutes=5),
        "owner": "data-team",
    },
)
def predictions_to_mongodb():

    # ─────────────────────────────────────────
    # Task 1 — Fetch all predictions from PostgreSQL
    # ─────────────────────────────────────────

    @task()
    def fetch_predictions() -> list:
        """
        Pull every row from stock_predictions.
        Casts dates and Decimals to plain Python types so they
        survive XCom serialisation and MongoDB insertion cleanly.
        """
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        rows = hook.get_records("""
            SELECT
                symbol,
                feature_date::text,
                prediction_date::text,
                predicted_direction,
                prob_up::float,
                prob_down::float,
                mlflow_run_id,
                dag_run_id,
                created_at::text
            FROM stock_predictions
            ORDER BY feature_date DESC, symbol ASC
        """)

        if not rows:
            log.warning("stock_predictions is empty — nothing to sync.")
            return []

        predictions = [
            {
                "symbol":               row[0],
                "feature_date":         row[1],
                "prediction_date":      row[2],
                "predicted_direction":  int(row[3]),
                "prob_up":              float(row[4]) if row[4] is not None else None,
                "prob_down":            float(row[5]) if row[5] is not None else None,
                "mlflow_run_id":        row[6],
                "dag_run_id":           row[7],
                "created_at":           row[8],
            }
            for row in rows
        ]

        log.info(f"Fetched {len(predictions)} predictions from PostgreSQL.")
        return predictions

    # ─────────────────────────────────────────
    # Task 2 — Upsert into MongoDB
    # ─────────────────────────────────────────

    @task()
    def push_to_mongodb(predictions: list):
        """
        Bulk-upsert all predictions into MongoDB stock_db.predictions.
        Uses (symbol, feature_date) as the unique key — re-running
        the DAG is safe and idempotent.

        Uses pymongo directly instead of MongoHook.get_conn() to avoid
        a known bug where MongoHook passes ssl=None to pymongo, which
        only accepts ssl=True or ssl=False.
        """
        from pymongo import MongoClient, UpdateOne
        from airflow.hooks.base import BaseHook

        if not predictions:
            log.warning("No predictions to push — skipping MongoDB write.")
            return

        # Read host/port/credentials from the Airflow connection
        # without going through MongoHook.get_conn() (which passes ssl=None)
        conn = BaseHook.get_connection(MONGO_CONN_ID)
        client = MongoClient(
            host=conn.host,
            port=int(conn.port or 27017),
            username=conn.login,
            password=conn.password,
            authSource="admin",
        )
        collection = client[MONGO_DB][MONGO_COLLECTION]

        # Ensure a unique index on (symbol, feature_date) for fast upserts
        collection.create_index(
            [("symbol", 1), ("feature_date", 1)],
            unique=True,
            background=True,
        )

        operations = [
            UpdateOne(
                filter={"symbol": p["symbol"], "feature_date": p["feature_date"]},
                update={"$set": p},
                upsert=True,
            )
            for p in predictions
        ]

        result = collection.bulk_write(operations, ordered=False)

        log.info(
            f"MongoDB sync complete — "
            f"upserted: {result.upserted_count} | "
            f"modified: {result.modified_count} | "
            f"total written: {len(predictions)}"
        )

    # ─────────────────────────────────────────
    # Task flow
    # ─────────────────────────────────────────

    predictions = fetch_predictions()
    push_to_mongodb(predictions)


predictions_to_mongodb()
