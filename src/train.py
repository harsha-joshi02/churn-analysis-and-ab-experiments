"""
XGBoost churn classifier:
  - Optuna hyperparameter search (n_trials configurable, default 20)
  - MLflow experiment tracking with full artefact logging
  - SHAP summary, ROC, PR, and confusion-matrix plots saved to disk + MLflow
  - Holdout predictions and full-population customer scores saved separately
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.features import (
    build_features,
    get_feature_columns,
    load_raw_data,
    prepare_train_data,
)
from src.utils import (
    CUSTOMER_SCORES_PATH,
    HOLDOUT_PREDICTIONS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    PLOTS_DIR,
    RAW_DATA_PATH,
    get_logger,
)

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Optuna objective ──────────────────────────────────────────────────────────

def _objective(
    trial: optuna.Trial,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    neg_pos_ratio: float = 1.0,
) -> float:
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        # search around the true neg/pos ratio so XGBoost weights positives correctly
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", neg_pos_ratio * 0.5, neg_pos_ratio * 2.0
        ),
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def tune_hyperparams(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 20,
    neg_pos_ratio: float = 1.0,
) -> dict[str, Any]:
    logger.info("Optuna HPO: %d trials …", n_trials)
    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_churn_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: _objective(t, X_tr, y_tr, X_val, y_val, neg_pos_ratio),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    logger.info(
        "Best AUC %.4f | params: %s", study.best_value, study.best_params
    )
    return study.best_params


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, auc: float, path: Path
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#4C72B0")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_pr_curve(
    y_true: np.ndarray, y_proba: np.ndarray, ap: float, path: Path
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#DD8452", lw=2, label=f"AP = {ap:.3f}")
    ax.fill_between(recall, precision, alpha=0.08, color="#DD8452")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_shap_summary(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    path: Path,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")


def _save_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    path: Path,
) -> None:
    importance = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=True)
        .tail(15)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi_df["feature"], fi_df["importance"], color="#4C72B0")
    ax.set_xlabel("Gain", fontsize=11)
    ax.set_title("Top 15 Feature Importances (XGBoost)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Main training pipeline ────────────────────────────────────────────────────

def train_pipeline(n_trials: int = 20) -> tuple[xgb.XGBClassifier, list[str], dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading raw data …")
    raw_df = load_raw_data(RAW_DATA_PATH)

    X_raw, y_raw = prepare_train_data(raw_df)
    feature_names = list(X_raw.columns)

    # Stratified splits: 60 / 20 / 20  (train / val / test)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )

    neg_pos_ratio = float((y_train == 0).sum()) / float((y_train == 1).sum())
    logger.info(
        "Splits — train: %d | val: %d | test: %d | neg/pos ratio: %.2f",
        len(X_train), len(X_val), len(X_test), neg_pos_ratio,
    )
    with mlflow.start_run(run_name="xgb_churn_training") as run:
        mlflow.set_tags(
            {
                "pipeline": "churn_training",
                "model_stage": "candidate",
                "target": "churn",
                "positive_class": "1",
                "split_strategy": "stratified_60_20_20",
                "split_random_state": "42",
            }
        )

        # Record which split was used for tuning, fitting, and evaluation.
        for context, X_split, y_split in (
            ("hyperparameter_training", X_train, y_train),
            ("hyperparameter_validation", X_val, y_val),
            ("training", X_trainval, y_trainval),
            ("evaluation", X_test, y_test),
        ):
            split_df = X_split.copy()
            split_df["churn"] = y_split
            dataset = mlflow.data.from_pandas(
                split_df,
                source=str(RAW_DATA_PATH),
                targets="churn",
                name=f"telco_churn_{context}",
            )
            mlflow.log_input(dataset, context=context)
        # ── HPO ───────────────────────────────────────────────────────────────
        mlflow.log_params(
            {
                "n_optuna_trials": n_trials,
                "smote": False,
                "neg_pos_ratio": round(neg_pos_ratio, 3),
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "n_features": len(feature_names),
            }
        )

        best_params = tune_hyperparams(X_train, y_train, X_val, y_val, n_trials, neg_pos_ratio)

        # ── Final model on train+val ──────────────────────────────────────────
        logger.info("Training final model …")
        final_params = {
            **best_params,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**final_params)
        model.fit(X_trainval, y_trainval, verbose=False)
        mlflow.log_params(best_params)

        # ── Evaluation ────────────────────────────────────────────────────────
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info("AUC-ROC %.4f | F1 %.4f | Avg-Precision %.4f", auc, f1, ap)
        mlflow.log_metrics({"auc_roc": auc, "f1_score": f1, "avg_precision": ap})

        # ── Plots → MLflow artefacts ──────────────────────────────────────────
        cm_path = PLOTS_DIR / "confusion_matrix.png"
        roc_path = PLOTS_DIR / "roc_curve.png"
        pr_path = PLOTS_DIR / "pr_curve.png"
        shap_path = PLOTS_DIR / "shap_summary.png"
        fi_path = PLOTS_DIR / "feature_importance.png"

        _save_confusion_matrix(cm, cm_path)
        _save_roc_curve(y_test.to_numpy(), y_proba, auc, roc_path)
        _save_pr_curve(y_test.to_numpy(), y_proba, ap, pr_path)

        shap_sample = X_test.sample(min(300, len(X_test)), random_state=42)
        _save_shap_summary(model, shap_sample, feature_names, shap_path)
        _save_feature_importance(model, feature_names, fi_path)

        for p in [cm_path, roc_path, pr_path, shap_path, fi_path]:
            mlflow.log_artifact(str(p), artifact_path="evaluation/plots")

        input_example = X_trainval.head(5)
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        # ── Persist model artefact ────────────────────────────────────────────
        with open(MODEL_PATH, "wb") as fh:
            pickle.dump(
                {
                    "model": model,
                    "feature_names": feature_names,
                    "mlflow_run_id": run.info.run_id,
                },
                fh,
            )
        logger.info("Model saved → %s", MODEL_PATH)

        # ── Evaluation predictions and full-customer scores ────────────────────
        # Keep unbiased evaluation rows distinct from full-population scores.
        raw_feat = build_features(raw_df)
        feat_cols = get_feature_columns(raw_feat)
        raw_X = raw_feat[feat_cols].astype(float)
        raw_proba = model.predict_proba(raw_X)[:, 1]

        holdout_df = raw_df.loc[X_test.index, ["customer_id"]].copy()
        holdout_df["y_true"] = y_test.to_numpy()
        holdout_df["churn_probability"] = y_proba
        holdout_df["y_pred"] = y_pred
        holdout_df["model_run_id"] = run.info.run_id

        customer_scores_df = raw_df.copy()
        for col in raw_feat.columns.difference(raw_df.columns):
            customer_scores_df[col] = raw_feat[col]
        customer_scores_df["churn_probability"] = raw_proba
        customer_scores_df["model_run_id"] = run.info.run_id

        HOLDOUT_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        holdout_df.to_parquet(HOLDOUT_PREDICTIONS_PATH, index=False)
        customer_scores_df.to_parquet(CUSTOMER_SCORES_PATH, index=False)
        mlflow.log_artifact(
            str(HOLDOUT_PREDICTIONS_PATH), artifact_path="evaluation/predictions"
        )
        mlflow.log_artifact(str(CUSTOMER_SCORES_PATH), artifact_path="scoring")
        logger.info("Holdout predictions saved → %s", HOLDOUT_PREDICTIONS_PATH)
        logger.info("Customer scores saved → %s", CUSTOMER_SCORES_PATH)

        metrics = {
            "auc_roc": auc,
            "f1_score": f1,
            "avg_precision": ap,
            "confusion_matrix": cm,
            "classification_report": report,
            "run_id": run.info.run_id,
            "feature_names": feature_names,
            "feature_importances": model.feature_importances_.tolist(),
        }

    return model, feature_names, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()
    train_pipeline(n_trials=args.n_trials)
