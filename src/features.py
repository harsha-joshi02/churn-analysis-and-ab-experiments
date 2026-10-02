"""Feature engineering pipeline for the IBM Telco Customer Churn dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import RAW_DATA_PATH, get_logger

logger = get_logger(__name__)

# ── Column names after normalisation ─────────────────────────────────────────
# Raw IBM dataset uses PascalCase / mixed; we rename to snake_case on load.
_RENAME_MAP = {
    "customerID":      "customer_id",
    "MonthlyCharges":  "monthly_charges",
    "TotalCharges":    "total_charges",
    "Contract":        "contract_type",
    "PaymentMethod":   "payment_method",
    "InternetService": "internet_service",
    "SeniorCitizen":   "senior_citizen",
    "Churn":           "churn",
}

_SERVICE_COLS = [
    "phone_service", "multiple_lines", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
]

# Fixed categorical encodings (consistent train / inference)
CATEGORICAL_MAPPINGS: dict[str, dict[str, int]] = {
    "contract_type": {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    },
    "payment_method": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3,
    },
    "internet_service": {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2,
    },
}

_NON_FEATURE_COLS: set[str] = {"customer_id", "churn"}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    # Rename to snake_case
    df = df.rename(columns=_RENAME_MAP)

    # Rename binary Yes/No columns to snake_case
    binary_rename = {
        "Partner": "partner",
        "Dependents": "dependents",
        "PhoneService": "phone_service",
        "PaperlessBilling": "paperless_billing",
        "MultipleLines": "multiple_lines",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
    }
    df = df.rename(columns=binary_rename)

    # Encode binary columns: "Yes" → 1, everything else → 0
    for col in list(binary_rename.values()):
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(np.int8)

    # gender: Male → 1, Female → 0
    if "gender" in df.columns:
        df["gender"] = (df["gender"] == "Male").astype(np.int8)

    # TotalCharges has blank strings for new customers (tenure=0) → fill with MonthlyCharges
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"])

    # Churn: "Yes" → 1, "No" → 0
    if not pd.api.types.is_numeric_dtype(df["churn"]):
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0}).astype(np.int8)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Number of active add-on services
    svc_cols = [c for c in _SERVICE_COLS if c in df.columns]
    df["num_products"] = df[svc_cols].sum(axis=1)

    # Revenue per product — high value, few products = churn risk
    df["charge_per_product"] = df["monthly_charges"] / (df["num_products"] + 1)

    # Interaction: long-tenure × high charges → sticky high-value customer
    df["tenure_x_charge"] = df["tenure"] * df["monthly_charges"] / 1000.0

    # Average monthly charge relative to total relationship value
    df["avg_monthly_over_total"] = df["monthly_charges"] / (df["total_charges"] + 1)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full feature engineering pipeline (no target required)."""
    df = encode_categoricals(df)
    df = create_derived_features(df)
    return df


def prepare_train_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for model training."""
    df = build_features(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].astype(float)
    y = df["churn"].astype(int)
    return X, y
