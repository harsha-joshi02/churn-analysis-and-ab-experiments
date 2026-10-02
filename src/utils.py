"""Shared configuration, logging, and path constants."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

BASE_DIR = Path(__file__).resolve().parent.parent


def get_logger(name: str) -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(name)-28s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://churn:churnpass@localhost:5432/churn_db",
)

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "churn_prediction",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "telco_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
HOLDOUT_PREDICTIONS_PATH = BASE_DIR / "models" / "holdout_predictions.parquet"
CUSTOMER_SCORES_PATH = BASE_DIR / "models" / "customer_scores.parquet"
PLOTS_DIR = BASE_DIR / "models" / "plots"
