"""Model inference, customer risk scoring, and segment-level summaries."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import shap

from src.features import build_features, load_raw_data
from src.utils import (
    CUSTOMER_SCORES_PATH,
    HOLDOUT_PREDICTIONS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    get_logger,
)

logger = get_logger(__name__)

# Probability thresholds that define the three risk tiers
RISK_BINS = [0.0, 0.30, 0.60, 1.01]
RISK_LABELS = ["Low", "Medium", "High"]


def load_model(path: Path = MODEL_PATH) -> tuple:
    """Return (model, feature_names) from the pickled artefact."""
    with open(path, "rb") as fh:
        art = pickle.load(fh)
    return art["model"], art["feature_names"]


def predict_churn(
    df: pd.DataFrame,
    model=None,
    feature_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Add `churn_probability` and `risk_segment` columns to *df*.
    Preserves all original columns.
    """
    if model is None:
        model, feature_names = load_model()

    df_feat = build_features(df.copy())
    X = df_feat[feature_names].astype(float)

    proba = model.predict_proba(X)[:, 1]
    result = df.copy()
    result["churn_probability"] = proba
    result["risk_segment"] = pd.cut(
        proba, bins=RISK_BINS, labels=RISK_LABELS, right=False
    )
    return result


def _load_scored_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "risk_segment" not in df.columns:
        df["risk_segment"] = pd.cut(
            df["churn_probability"], bins=RISK_BINS, labels=RISK_LABELS, right=False
        )
    return df


def load_customer_scores(path: Path = CUSTOMER_SCORES_PATH) -> pd.DataFrame:
    """Load scores for the complete customer population."""
    return _load_scored_data(path)


def load_holdout_predictions(
    path: Path = HOLDOUT_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Load predictions produced only for the untouched evaluation holdout."""
    return _load_scored_data(path)


def load_predictions(path: Path = CUSTOMER_SCORES_PATH) -> pd.DataFrame:
    """Backward-compatible alias for :func:`load_customer_scores`."""
    return load_customer_scores(path)


def segment_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregated KPIs per risk segment."""
    return (
        predictions.groupby("risk_segment", observed=True)
        .agg(
            count=("customer_id", "count"),
            avg_churn_prob=("churn_probability", "mean"),
            avg_tenure=("tenure", "mean"),
            avg_monthly_charges=("monthly_charges", "mean"),
        )
        .reset_index()
        .sort_values("avg_churn_prob", ascending=False)
    )


def get_top_churn_drivers(
    df: pd.DataFrame,
    customer_id: str,
    model=None,
    feature_names: Optional[list[str]] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    SHAP-based top-N churn drivers for a single customer.
    Returns a DataFrame with columns: feature, shap_value, feature_value.
    """
    if model is None:
        model, feature_names = load_model()

    row = df[df["customer_id"] == customer_id]
    if row.empty:
        raise ValueError(f"Customer {customer_id!r} not found.")

    df_feat = build_features(row.copy())
    X = df_feat[feature_names].astype(float)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    drivers = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_vals[0],
            "feature_value": X.iloc[0].values,
        }
    )
    drivers["abs_shap"] = drivers["shap_value"].abs()
    return (
        drivers.sort_values("abs_shap", ascending=False)
        .head(top_n)
        .drop(columns="abs_shap")
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    df = load_raw_data(RAW_DATA_PATH)
    preds = predict_churn(df)
    print(segment_summary(preds).to_string(index=False))
