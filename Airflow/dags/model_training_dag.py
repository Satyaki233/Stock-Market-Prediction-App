"""
Indian Stock Market - Model Training DAG
Airflow 3.x (uses airflow.sdk imports)

Schedule : Daily at 6:00 PM IST (12:30 UTC) — 1 hour after feature engineering (11:30 UTC)
Source   : stock_features
Outputs  : stock_predictions  (Postgres)
           MLflow experiment  (http://mlflow:5000)

Approach
────────
  Model   : LightGBM binary classifier
  Target  : target_direction_1d  (1 = next-day close > today's close, 0 = down)
  Scope   : Global model — trained on ALL symbols simultaneously
  CV      : 3-fold walk-forward (chronological splits, no data leakage)
  Final   : Retrain on full labelled history → register in MLflow
  Predict : Latest available feature row per symbol → stock_predictions

Task flow
─────────
  validate_model_tables
          ↓
  load_training_data   → returns date-range stats via XCom
          ↓
  walk_forward_train   → 3-fold CV, logs avg metrics to MLflow
          ↓
  train_final_model    → full retrain, logs model artifact, returns mlflow_run_id
          ↓
  generate_predictions → loads model from MLflow, predicts latest row per symbol
          ↓
  model_summary
"""

import logging
import os

import pendulum
from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

log = logging.getLogger(__name__)

POSTGRES_CONN_ID   = "stock_db_conn"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT  = "stock_direction_prediction"

# ── Feature columns fed into the model ────────────────────────────────────────
# These must all exist in stock_features.
# Sector / industry are treated as LightGBM categorical features (no encoding needed).
NUMERIC_FEATURES = [
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
]
CATEGORICAL_FEATURES = ["sector", "industry"]
ALL_FEATURES  = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL    = "target_direction_1d"

# LightGBM hyperparameters
LGB_PARAMS = {
    "objective":        "binary",
    "metric":           ["binary_logloss", "auc"],
    "num_leaves":       63,
    "max_depth":        6,
    "learning_rate":    0.05,
    "n_estimators":     1000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 50,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":          -1,
    "n_jobs":           -1,
    "random_state":     42,
}


# ─────────────────────────────────────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────────────────────────────────────

@dag(
    dag_id="stock_model_training",
    description="Train LightGBM direction classifier on stock_features; store predictions",
    schedule="30 12 * * 1-5",          # 12:30 UTC = 6:00 PM IST, weekdays
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["stocks", "ml", "lgbm", "prediction"],
    default_args={
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=10),
        "owner": "data-team",
    },
)
def stock_model_training():

    # ─────────────────────────────────────────
    # Task 1 — Create DB tables
    # ─────────────────────────────────────────

    @task()
    def validate_model_tables():
        """Create model_runs and stock_predictions tables if they don't exist."""
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        hook.run("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id               SERIAL PRIMARY KEY,
                mlflow_run_id    VARCHAR(100),
                model_version    VARCHAR(50),
                train_from       DATE,
                train_to         DATE,
                val_from         DATE,
                val_to           DATE,
                n_train_rows     INTEGER,
                n_val_rows       INTEGER,
                n_features       INTEGER,
                -- walk-forward avg metrics
                cv_accuracy      NUMERIC(8,6),
                cv_auc_roc       NUMERIC(8,6),
                cv_f1            NUMERIC(8,6),
                cv_log_loss      NUMERIC(10,6),
                -- final validation metrics
                val_accuracy     NUMERIC(8,6),
                val_auc_roc      NUMERIC(8,6),
                val_f1           NUMERIC(8,6),
                val_log_loss     NUMERIC(10,6),
                dag_run_id       VARCHAR(100),
                created_at       TIMESTAMP DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS stock_predictions (
                id                   SERIAL PRIMARY KEY,
                symbol               VARCHAR(30)  NOT NULL,
                feature_date         DATE         NOT NULL,  -- date of features used (already traded)
                prediction_date      DATE         NOT NULL,  -- date being predicted (next trading day)
                predicted_direction  SMALLINT     NOT NULL,  -- 1=up, 0=down
                prob_up              NUMERIC(8,6),           -- P(direction=1)
                prob_down            NUMERIC(8,6),           -- P(direction=0)
                mlflow_run_id        VARCHAR(100),
                dag_run_id           VARCHAR(100),
                created_at           TIMESTAMP    DEFAULT NOW(),
                UNIQUE (symbol, feature_date)
            );

            CREATE INDEX IF NOT EXISTS idx_pred_symbol_date
                ON stock_predictions(symbol, prediction_date DESC);
            CREATE INDEX IF NOT EXISTS idx_pred_date
                ON stock_predictions(prediction_date DESC);

            -- add prediction_date to existing table if this is an upgrade
            ALTER TABLE stock_predictions
                ADD COLUMN IF NOT EXISTS prediction_date DATE;
        """)
        log.info("model_runs and stock_predictions tables ready.")
        return "tables_ready"

    # ─────────────────────────────────────────
    # Task 2 — Load & validate training data
    # ─────────────────────────────────────────

    @task()
    def load_training_data(_: str) -> dict:
        """
        Pull all labelled rows from stock_features.
        Returns summary stats (date range, row count) via XCom — the actual
        DataFrame is not passed through XCom; each downstream task re-queries.
        """
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        stats = hook.get_first("""
            SELECT
                COUNT(*)                                  AS n_rows,
                COUNT(DISTINCT symbol)                    AS n_symbols,
                MIN(date)::text                           AS date_min,
                MAX(date)::text                           AS date_max,
                SUM(CASE WHEN target_direction_1d = 1
                          THEN 1 ELSE 0 END)::float
                    / NULLIF(COUNT(*), 0)                 AS pct_up
            FROM stock_features
            WHERE target_direction_1d IS NOT NULL
        """)

        n_rows, n_symbols, date_min, date_max, pct_up = stats

        if n_rows < 1000:
            raise ValueError(
                f"Insufficient training data: {n_rows} rows. "
                "Run ingestion + feature DAGs first."
            )

        log.info(
            f"Training data: {n_rows:,} rows | {n_symbols} symbols | "
            f"{date_min} → {date_max} | {pct_up:.1%} up days"
        )
        return {
            "n_rows":    int(n_rows),
            "n_symbols": int(n_symbols),
            "date_min":  date_min,
            "date_max":  date_max,
            "pct_up":    float(pct_up),
        }

    # ─────────────────────────────────────────
    # Task 3 — Walk-forward cross-validation
    # ─────────────────────────────────────────

    @task()
    def walk_forward_train(data_stats: dict, dag_run_id: str) -> dict:
        """
        3-fold chronological walk-forward cross-validation.

        Fold structure (by sorted unique dates):
          Fold 1 — train: first 60%,  val: 60%–70%
          Fold 2 — train: first 70%,  val: 70%–80%
          Fold 3 — train: first 80%,  val: 80%–90%

        The last 10% of dates are held out for the final model evaluation
        (train_final_model task).

        Logs per-fold and average metrics to MLflow.
        Returns average CV metrics dict.
        """
        import numpy as np
        import pandas as pd
        import lightgbm as lgb
        import mlflow
        import mlflow.lightgbm
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score, log_loss
        )
        from sklearn.preprocessing import LabelEncoder

        hook   = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        engine = hook.get_sqlalchemy_engine()

        # ── Load full labelled dataset ────────────────────────────────────
        select_cols = ", ".join(["symbol", "date", TARGET_COL] + ALL_FEATURES)
        df = pd.read_sql(
            f"""
            SELECT {select_cols}
            FROM   stock_features
            WHERE  target_direction_1d IS NOT NULL
            ORDER  BY date ASC
            """,
            engine,
        )

        df["date"] = pd.to_datetime(df["date"])

        # ── Label-encode categoricals ─────────────────────────────────────
        le_sector   = LabelEncoder()
        le_industry = LabelEncoder()
        df["sector"]   = le_sector.fit_transform(df["sector"].fillna("Unknown"))
        df["industry"] = le_industry.fit_transform(df["industry"].fillna("Unknown"))

        # ── Build fold date boundaries ────────────────────────────────────
        sorted_dates = sorted(df["date"].unique())
        n = len(sorted_dates)

        fold_boundaries = [
            (0.00, 0.60, 0.70),   # Fold 1
            (0.00, 0.70, 0.80),   # Fold 2
            (0.00, 0.80, 0.90),   # Fold 3
        ]

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        fold_metrics = []

        with mlflow.start_run(run_name=f"walk_forward_cv_{dag_run_id}") as cv_run:
            mlflow.log_params({**LGB_PARAMS, "n_folds": 3, "target": TARGET_COL})
            mlflow.log_param("n_total_rows",   data_stats["n_rows"])
            mlflow.log_param("n_symbols",      data_stats["n_symbols"])
            mlflow.log_param("features",       ",".join(ALL_FEATURES))

            for fold_idx, (train_start_pct, train_end_pct, val_end_pct) in enumerate(fold_boundaries, 1):
                train_end_date = sorted_dates[int(n * train_end_pct) - 1]
                val_end_date   = sorted_dates[int(n * val_end_pct)   - 1]

                train_mask = df["date"] <= train_end_date
                val_mask   = (df["date"] > train_end_date) & (df["date"] <= val_end_date)

                X_train = df.loc[train_mask, ALL_FEATURES]
                y_train = df.loc[train_mask, TARGET_COL].astype(int)
                X_val   = df.loc[val_mask,   ALL_FEATURES]
                y_val   = df.loc[val_mask,   TARGET_COL].astype(int)

                model = lgb.LGBMClassifier(**LGB_PARAMS)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                    categorical_feature=CATEGORICAL_FEATURES,
                )

                y_pred      = model.predict(X_val)
                y_prob      = model.predict_proba(X_val)[:, 1]

                metrics = {
                    "accuracy":  accuracy_score(y_val, y_pred),
                    "auc_roc":   roc_auc_score(y_val, y_prob),
                    "f1":        f1_score(y_val, y_pred, zero_division=0),
                    "log_loss":  log_loss(y_val, y_prob),
                }
                fold_metrics.append(metrics)

                for k, v in metrics.items():
                    mlflow.log_metric(f"fold{fold_idx}_{k}", v)

                log.info(
                    f"  Fold {fold_idx}: acc={metrics['accuracy']:.4f} "
                    f"auc={metrics['auc_roc']:.4f} f1={metrics['f1']:.4f} "
                    f"loss={metrics['log_loss']:.4f} "
                    f"| train≤{train_end_date.date()} val≤{val_end_date.date()} "
                    f"| best_iter={model.best_iteration_}"
                )

            # Average CV metrics
            avg = {
                k: float(np.mean([m[k] for m in fold_metrics]))
                for k in fold_metrics[0]
            }
            for k, v in avg.items():
                mlflow.log_metric(f"cv_avg_{k}", v)

            log.info(
                f"Walk-forward CV — avg: acc={avg['accuracy']:.4f} "
                f"auc={avg['auc_roc']:.4f} f1={avg['f1']:.4f} loss={avg['log_loss']:.4f}"
            )

        return avg

    # ─────────────────────────────────────────
    # Task 4 — Train final production model
    # ─────────────────────────────────────────

    @task()
    def train_final_model(cv_metrics: dict, data_stats: dict, dag_run_id: str) -> str:
        """
        Train on 90% of labelled history (oldest dates), validate on the last 10%.
        Logs the trained model and feature importances to MLflow.
        Returns the MLflow run_id so generate_predictions can load the model.
        """
        import numpy as np
        import pandas as pd
        import lightgbm as lgb
        import mlflow
        import mlflow.lightgbm
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score, log_loss,
            classification_report,
        )
        from sklearn.preprocessing import LabelEncoder

        hook   = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        engine = hook.get_sqlalchemy_engine()

        # ── Load full dataset ─────────────────────────────────────────────
        select_cols = ", ".join(["symbol", "date", TARGET_COL] + ALL_FEATURES)
        df = pd.read_sql(
            f"""
            SELECT {select_cols}
            FROM   stock_features
            WHERE  target_direction_1d IS NOT NULL
            ORDER  BY date ASC
            """,
            engine,
        )

        df["date"] = pd.to_datetime(df["date"])

        # ── Encode categoricals ───────────────────────────────────────────
        le_sector   = LabelEncoder()
        le_industry = LabelEncoder()
        df["sector"]   = le_sector.fit_transform(df["sector"].fillna("Unknown"))
        df["industry"] = le_industry.fit_transform(df["industry"].fillna("Unknown"))

        # ── 90 / 10 chronological split ───────────────────────────────────
        sorted_dates   = sorted(df["date"].unique())
        cutoff_date    = sorted_dates[int(len(sorted_dates) * 0.90) - 1]

        train_mask = df["date"] <= cutoff_date
        val_mask   = df["date"] >  cutoff_date

        X_train = df.loc[train_mask, ALL_FEATURES]
        y_train = df.loc[train_mask, TARGET_COL].astype(int)
        X_val   = df.loc[val_mask,   ALL_FEATURES]
        y_val   = df.loc[val_mask,   TARGET_COL].astype(int)

        log.info(
            f"Final model — train: {len(X_train):,} rows (≤{cutoff_date.date()}) | "
            f"val: {len(X_val):,} rows"
        )

        # ── Train ─────────────────────────────────────────────────────────
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=f"final_model_{dag_run_id}") as run:
            model = lgb.LGBMClassifier(**LGB_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
                categorical_feature=CATEGORICAL_FEATURES,
            )

            # ── Evaluate ──────────────────────────────────────────────────
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            val_metrics = {
                "val_accuracy": accuracy_score(y_val, y_pred),
                "val_auc_roc":  roc_auc_score(y_val, y_prob),
                "val_f1":       f1_score(y_val, y_pred, zero_division=0),
                "val_log_loss": log_loss(y_val, y_prob),
            }

            mlflow.log_params({**LGB_PARAMS, "best_iteration": model.best_iteration_})
            mlflow.log_metrics(val_metrics)
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
            mlflow.log_param("train_cutoff_date", str(cutoff_date.date()))
            mlflow.log_param("n_features",        len(ALL_FEATURES))
            mlflow.log_param("features",          ",".join(ALL_FEATURES))
            mlflow.log_param("dag_run_id",        dag_run_id)

            # ── Feature importance ────────────────────────────────────────
            importance = pd.DataFrame({
                "feature":    ALL_FEATURES,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            log.info(f"Top 10 features:\n{importance.head(10).to_string(index=False)}")

            # Log importance as a CSV artifact
            import tempfile, os
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp:
                importance.to_csv(tmp, index=False)
                tmp_path = tmp.name
            mlflow.log_artifact(tmp_path, artifact_path="feature_importance")
            os.unlink(tmp_path)

            # ── Log model artifact ────────────────────────────────────────
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="model",
                registered_model_name="stock_direction_lgbm",
            )

            mlflow_run_id = run.info.run_id

        # ── Persist run metadata to Postgres ─────────────────────────────
        hook.run("""
            INSERT INTO model_runs
                (mlflow_run_id, model_version,
                 train_from, train_to, val_from, val_to,
                 n_train_rows, n_val_rows, n_features,
                 cv_accuracy, cv_auc_roc, cv_f1, cv_log_loss,
                 val_accuracy, val_auc_roc, val_f1, val_log_loss,
                 dag_run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, parameters=(
            mlflow_run_id,
            f"run_{dag_run_id[:8]}",
            data_stats["date_min"], str(cutoff_date.date()),
            str(sorted_dates[int(len(sorted_dates) * 0.90)].date()),
            data_stats["date_max"],
            int(len(X_train)), int(len(X_val)), len(ALL_FEATURES),
            cv_metrics.get("accuracy"), cv_metrics.get("auc_roc"),
            cv_metrics.get("f1"),       cv_metrics.get("log_loss"),
            val_metrics["val_accuracy"], val_metrics["val_auc_roc"],
            val_metrics["val_f1"],       val_metrics["val_log_loss"],
            dag_run_id,
        ))

        log.info(
            f"Final model — val acc={val_metrics['val_accuracy']:.4f} "
            f"auc={val_metrics['val_auc_roc']:.4f} "
            f"f1={val_metrics['val_f1']:.4f} | MLflow run: {mlflow_run_id}"
        )
        return mlflow_run_id

    # ─────────────────────────────────────────
    # Task 5 — Generate predictions
    # ─────────────────────────────────────────

    @task()
    def generate_predictions(mlflow_run_id: str, dag_run_id: str):
        """
        Load the trained model from MLflow.
        For each symbol fetch its most recent feature row from stock_features.
        Predict direction + probability and upsert into stock_predictions.
        """
        import pandas as pd
        import mlflow.lightgbm
        from pandas.tseries.offsets import BDay
        from sklearn.preprocessing import LabelEncoder

        hook   = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        engine = hook.get_sqlalchemy_engine()

        # ── Load model from MLflow ────────────────────────────────────────
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"runs:/{mlflow_run_id}/model"
        model     = mlflow.lightgbm.load_model(model_uri)
        log.info(f"Loaded model from MLflow run: {mlflow_run_id}")

        # ── Fetch latest feature row per symbol ───────────────────────────
        select_cols = ", ".join(["symbol", "date"] + ALL_FEATURES)
        latest_df   = pd.read_sql(
            f"""
            SELECT DISTINCT ON (symbol) {select_cols}
            FROM   stock_features
            ORDER  BY symbol, date DESC
            """,
            engine,
        )

        if latest_df.empty:
            log.warning("No rows in stock_features — skipping predictions.")
            return

        log.info(f"Generating predictions for {len(latest_df)} symbols.")

        # ── Encode categoricals (same logic as training) ──────────────────
        le_sector   = LabelEncoder()
        le_industry = LabelEncoder()

        # Fit on all known values from training data to avoid unseen-label issues
        all_sectors    = pd.read_sql(
            "SELECT DISTINCT sector   FROM stock_features", engine
        )["sector"].fillna("Unknown").tolist()
        all_industries = pd.read_sql(
            "SELECT DISTINCT industry FROM stock_features", engine
        )["industry"].fillna("Unknown").tolist()

        le_sector.fit(all_sectors)
        le_industry.fit(all_industries)

        latest_df["sector"]   = latest_df["sector"].fillna("Unknown").map(
            lambda x: x if x in le_sector.classes_ else "Unknown"
        )
        latest_df["industry"] = latest_df["industry"].fillna("Unknown").map(
            lambda x: x if x in le_industry.classes_ else "Unknown"
        )
        latest_df["sector"]   = le_sector.transform(latest_df["sector"])
        latest_df["industry"] = le_industry.transform(latest_df["industry"])

        # ── Predict ───────────────────────────────────────────────────────
        X          = latest_df[ALL_FEATURES]
        probs      = model.predict_proba(X)   # shape (n, 2): [P(down), P(up)]
        directions = (probs[:, 1] >= 0.5).astype(int)

        latest_df["predicted_direction"] = directions
        latest_df["prob_up"]             = probs[:, 1]
        latest_df["prob_down"]           = probs[:, 0]
        # next business day after feature_date = the day being predicted
        latest_df["prediction_date"] = pd.to_datetime(latest_df["date"]) + BDay(1)
        latest_df["prediction_date"] = latest_df["prediction_date"].dt.date

        # ── Upsert into stock_predictions ─────────────────────────────────
        rows = [
            (
                row["symbol"],
                row["date"],
                row["prediction_date"],
                int(row["predicted_direction"]),
                float(row["prob_up"]),
                float(row["prob_down"]),
                mlflow_run_id,
                dag_run_id,
            )
            for _, row in latest_df.iterrows()
        ]

        hook.insert_rows(
            table="stock_predictions",
            rows=rows,
            target_fields=[
                "symbol", "feature_date", "prediction_date",
                "predicted_direction", "prob_up", "prob_down",
                "mlflow_run_id", "dag_run_id",
            ],
            replace=True,
            replace_index=["symbol", "feature_date"],
        )

        up_count   = int((directions == 1).sum())
        down_count = int((directions == 0).sum())
        log.info(
            f"Predictions stored — UP: {up_count} | DOWN: {down_count} | "
            f"Total: {len(rows)}"
        )

    # ─────────────────────────────────────────
    # Task 6 — Summary
    # ─────────────────────────────────────────

    @task()
    def model_summary(dag_run_id: str):
        """Log a final summary of the training run and stored predictions."""
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

        run_row = hook.get_first("""
            SELECT mlflow_run_id, val_accuracy, val_auc_roc, val_f1,
                   cv_accuracy, cv_auc_roc, n_train_rows, n_val_rows
            FROM   model_runs
            WHERE  dag_run_id = %s
            ORDER  BY created_at DESC
            LIMIT  1
        """, parameters=(dag_run_id,))

        pred_row = hook.get_first("""
            SELECT COUNT(*),
                   SUM(CASE WHEN predicted_direction = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN predicted_direction = 0 THEN 1 ELSE 0 END)
            FROM   stock_predictions
            WHERE  dag_run_id = %s
        """, parameters=(dag_run_id,))

        log.info("=" * 60)
        log.info(f"MODEL TRAINING SUMMARY — Run: {dag_run_id}")
        log.info("=" * 60)
        if run_row:
            mlflow_id, val_acc, val_auc, val_f1, cv_acc, cv_auc, n_train, n_val = run_row
            log.info(f"  MLflow run     : {mlflow_id}")
            log.info(f"  Train rows     : {n_train:,} | Val rows: {n_val:,}")
            log.info(f"  CV  acc={cv_acc:.4f}  auc={cv_auc:.4f}")
            log.info(f"  Val acc={val_acc:.4f}  auc={val_auc:.4f}  f1={val_f1:.4f}")
        if pred_row:
            total, n_up, n_down = pred_row
            log.info(f"  Predictions    : {total} total | UP={n_up} | DOWN={n_down}")
        log.info("=" * 60)

    # ─────────────────────────────────────────
    # Task flow
    # ─────────────────────────────────────────

    dag_run_id   = "{{ run_id }}"

    tables_ready  = validate_model_tables()
    data_stats    = load_training_data(tables_ready)
    cv_metrics    = walk_forward_train(data_stats=data_stats, dag_run_id=dag_run_id)
    mlflow_run_id = train_final_model(
        cv_metrics=cv_metrics, data_stats=data_stats, dag_run_id=dag_run_id
    )
    preds_done    = generate_predictions(
        mlflow_run_id=mlflow_run_id, dag_run_id=dag_run_id
    )
    preds_done >> model_summary(dag_run_id=dag_run_id)


# Register the DAG
stock_model_training()
