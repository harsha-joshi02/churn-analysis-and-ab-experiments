"""
Bayesian A/B testing engine.

Design rationale
────────────────
We model retention (1 − churn) as the success probability in a Bernoulli
process.  The conjugate prior for a Bernoulli likelihood is the Beta
distribution: Beta(α, β).

  Posterior after n trials / k successes:
    Beta(α_prior + k, β_prior + n - k)

Advantages over frequentist NHST:
  - No hard p-value threshold; instead we compute P(treatment > control)
    continuously as data accumulates.
  - Naturally incorporates prior knowledge (e.g. historical retention rate).
  - Decision is probabilistic: "treatment wins with 92% probability" is more
    actionable than "p = 0.04".
  - Expected loss / lift gives business-ready numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta as beta_dist

from src.utils import get_logger

logger = get_logger(__name__)

_N_MC = 60_000  # Monte-Carlo samples for posterior integrals


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExperimentData:
    name: str
    description: str
    segment: str
    control_description: str
    treatment_description: str
    control_conversions: int = 0    # "successes" = retained customers
    control_trials: int = 0
    treatment_conversions: int = 0
    treatment_trials: int = 0
    prior_alpha: float = 1.0        # uninformative prior Beta(1,1) = Uniform
    prior_beta: float = 1.0


@dataclass
class BayesianResult:
    prob_treatment_beats_control: float
    expected_lift: float
    recommended_sample_size: int
    control_rate: float
    treatment_rate: float
    control_posterior: dict[str, Any]      # alpha, beta, samples list
    treatment_posterior: dict[str, Any]


def _validate_beta_params(alpha: float, beta: float, label: str) -> None:
    if isinstance(alpha, (bool, np.bool_)) or isinstance(beta, (bool, np.bool_)):
        raise ValueError(f"{label} alpha and beta must be finite and greater than zero.")
    try:
        valid = np.isfinite(alpha) and np.isfinite(beta) and alpha > 0 and beta > 0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{label} alpha and beta must be finite and greater than zero.")


def validate_experiment(exp: ExperimentData) -> None:
    """Validate counts and prior parameters before any posterior calculation."""
    if not isinstance(exp.name, str) or not exp.name.strip():
        raise ValueError("Experiment name must not be empty.")

    for arm, conversions, trials in (
        ("control", exp.control_conversions, exp.control_trials),
        ("treatment", exp.treatment_conversions, exp.treatment_trials),
    ):
        if (
            isinstance(conversions, (bool, np.bool_))
            or not isinstance(conversions, (int, np.integer))
            or isinstance(trials, (bool, np.bool_))
            or not isinstance(trials, (int, np.integer))
        ):
            raise TypeError(f"{arm} conversions and trials must be integers.")
        if conversions < 0 or trials < 0:
            raise ValueError(f"{arm} conversions and trials must be non-negative.")
        if conversions > trials:
            raise ValueError(f"{arm} conversions cannot exceed trials.")

    _validate_beta_params(exp.prior_alpha, exp.prior_beta, "Prior")


# ── Core Bayesian maths ───────────────────────────────────────────────────────

def _posterior_params(
    conversions: int,
    trials: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[float, float]:
    """Return (α_post, β_post) of the Beta posterior."""
    _validate_beta_params(prior_alpha, prior_beta, "Prior")
    if conversions < 0 or trials < 0 or conversions > trials:
        raise ValueError("Conversions must be between zero and trials.")
    return prior_alpha + conversions, prior_beta + (trials - conversions)


def prob_b_beats_a(
    alpha_a: float,
    beta_a: float,
    alpha_b: float,
    beta_b: float,
    n_samples: int = _N_MC,
    seed: int = 0,
) -> float:
    """Monte-Carlo estimate of P(B > A)."""
    _validate_beta_params(alpha_a, beta_a, "Control posterior")
    _validate_beta_params(alpha_b, beta_b, "Treatment posterior")
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")
    rng = np.random.default_rng(seed)
    a = beta_dist.rvs(alpha_a, beta_a, size=n_samples, random_state=rng)
    b = beta_dist.rvs(alpha_b, beta_b, size=n_samples, random_state=rng)
    return float(np.mean(b > a))


def expected_lift(
    alpha_a: float,
    beta_a: float,
    alpha_b: float,
    beta_b: float,
    n_samples: int = _N_MC,
    seed: int = 1,
) -> float:
    """Relative lift between posterior means.

    This is stable even when a valid control posterior has mass close to zero;
    averaging ``(B - A) / A`` samples is not (and can have infinite expectation).
    ``n_samples`` and ``seed`` remain accepted for API compatibility.
    """
    del n_samples, seed
    _validate_beta_params(alpha_a, beta_a, "Control posterior")
    _validate_beta_params(alpha_b, beta_b, "Treatment posterior")
    control_mean = alpha_a / (alpha_a + beta_a)
    treatment_mean = alpha_b / (alpha_b + beta_b)
    return float((treatment_mean - control_mean) / control_mean)


def recommend_sample_size(
    baseline_rate: float,
    mde: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Minimum sample size per arm using the two-proportion z-test formula
    (conservative approximation for Bayesian context).
    """
    baseline_rate = np.clip(baseline_rate, 0.01, 0.99)
    p1 = baseline_rate
    p2 = min(baseline_rate + mde, 0.999)
    p_bar = (p1 + p2) / 2.0
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    z_b = stats.norm.ppf(power)
    numerator = (
        z_a * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    return int(np.ceil(numerator / denominator))


def run_bayesian_ab_test(exp: ExperimentData) -> BayesianResult:
    """Full Bayesian A/B analysis from an ExperimentData object."""
    validate_experiment(exp)
    a_alpha, a_beta = _posterior_params(
        exp.control_conversions, exp.control_trials,
        exp.prior_alpha, exp.prior_beta,
    )
    b_alpha, b_beta = _posterior_params(
        exp.treatment_conversions, exp.treatment_trials,
        exp.prior_alpha, exp.prior_beta,
    )

    p_win = prob_b_beats_a(a_alpha, a_beta, b_alpha, b_beta)
    lift = expected_lift(a_alpha, a_beta, b_alpha, b_beta)

    baseline = (
        exp.control_conversions / exp.control_trials
        if exp.control_trials > 0 else 0.20
    )
    n_recommend = recommend_sample_size(baseline)

    n_post_samples = 500
    rng = np.random.default_rng(99)
    ctrl_samples = beta_dist.rvs(a_alpha, a_beta, size=n_post_samples, random_state=rng).tolist()
    trt_samples = beta_dist.rvs(b_alpha, b_beta, size=n_post_samples, random_state=rng).tolist()

    return BayesianResult(
        prob_treatment_beats_control=p_win,
        expected_lift=lift,
        recommended_sample_size=n_recommend,
        control_rate=exp.control_conversions / max(exp.control_trials, 1),
        treatment_rate=exp.treatment_conversions / max(exp.treatment_trials, 1),
        control_posterior={"alpha": a_alpha, "beta": a_beta, "samples": ctrl_samples},
        treatment_posterior={"alpha": b_alpha, "beta": b_beta, "samples": trt_samples},
    )


# ── Simulation helper ─────────────────────────────────────────────────────────

def simulate_discount_experiment(
    high_risk_df: pd.DataFrame,
    discount_lift_pct: float = 0.15,
    seed: int = 42,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[ExperimentData, BayesianResult]:
    """
    Simulate a 20%-discount intervention on the High-risk cohort.

    Control  : no action (baseline retention = 1 − avg_churn_probability)
    Treatment: discount offer raises retention by `discount_lift_pct`
    """
    if high_risk_df.empty:
        raise ValueError("Experiment cohort must not be empty.")
    if "churn_probability" not in high_risk_df.columns:
        raise ValueError("Experiment cohort must contain churn_probability.")
    churn_probabilities = high_risk_df["churn_probability"]
    if (
        churn_probabilities.isna().any()
        or not churn_probabilities.between(0, 1).all()
    ):
        raise ValueError("churn_probability values must be between 0 and 1.")
    if not np.isfinite(discount_lift_pct) or not 0 <= discount_lift_pct <= 1:
        raise ValueError("discount_lift_pct must be between 0 and 1.")
    _validate_beta_params(prior_alpha, prior_beta, "Prior")

    n = len(high_risk_df)
    n_ctrl = n // 2
    n_trt = n - n_ctrl

    avg_churn = high_risk_df["churn_probability"].mean()
    p_ctrl = float(np.clip(1.0 - avg_churn, 0.01, 0.99))
    p_trt = float(np.clip(p_ctrl + discount_lift_pct, 0.01, 0.99))

    rng = np.random.default_rng(seed)
    ctrl_conversions = int(rng.binomial(n_ctrl, p_ctrl))
    trt_conversions = int(rng.binomial(n_trt, p_trt))

    exp = ExperimentData(
        name="discount_intervention_high_risk",
        description=(
            "20 % monthly discount for 3 months offered to High-risk customers "
            "vs no intervention (control)."
        ),
        segment="High",
        control_description="No intervention",
        treatment_description="20% discount for 3 months",
        control_conversions=ctrl_conversions,
        control_trials=n_ctrl,
        treatment_conversions=trt_conversions,
        treatment_trials=n_trt,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )
    result = run_bayesian_ab_test(exp)
    logger.info(
        "Simulated experiment | ctrl_rate=%.3f | trt_rate=%.3f | "
        "P(trt>ctrl)=%.3f | lift=%.1f%%",
        result.control_rate,
        result.treatment_rate,
        result.prob_treatment_beats_control,
        result.expected_lift * 100,
    )
    return exp, result


# ── Database persistence ──────────────────────────────────────────────────────

def save_experiment_to_db(exp: ExperimentData, result: BayesianResult) -> None:
    from src.db import ExperimentConfig, ExperimentResult, SessionLocal, is_db_available

    if not is_db_available():
        logger.warning("DB unavailable — experiment not persisted.")
        return

    db = SessionLocal()
    try:
        cfg = ExperimentConfig(
            name=exp.name,
            description=exp.description,
            segment=exp.segment,
            control_description=exp.control_description,
            treatment_description=exp.treatment_description,
            config_json=asdict(exp),
        )
        # Upsert config
        existing = db.query(ExperimentConfig).filter_by(name=exp.name).first()
        if existing:
            existing.description = exp.description
            existing.segment = exp.segment
            existing.control_description = exp.control_description
            existing.treatment_description = exp.treatment_description
            existing.config_json = asdict(exp)
            existing.created_at = datetime.utcnow()
        else:
            db.add(cfg)

        res = ExperimentResult(
            experiment_name=exp.name,
            control_conversions=exp.control_conversions,
            control_trials=exp.control_trials,
            treatment_conversions=exp.treatment_conversions,
            treatment_trials=exp.treatment_trials,
            prob_treatment_beats_control=result.prob_treatment_beats_control,
            expected_lift=result.expected_lift,
            recommended_sample_size=result.recommended_sample_size,
            posterior_data={
                "control_samples": result.control_posterior["samples"],
                "treatment_samples": result.treatment_posterior["samples"],
            },
        )
        db.add(res)
        db.commit()
        logger.info("Saved experiment '%s' to database.", exp.name)
    except Exception as exc:
        db.rollback()
        logger.error("DB save failed: %s", exc)
        raise
    finally:
        db.close()


def get_all_experiments() -> list[dict]:
    from src.db import ExperimentConfig, ExperimentResult, SessionLocal, is_db_available

    if not is_db_available():
        return []

    db = SessionLocal()
    try:
        results = (
            db.query(ExperimentResult)
            .order_by(ExperimentResult.created_at.desc())
            .all()
        )
        configs = {
            c.name: c
            for c in db.query(ExperimentConfig).all()
        }
        out = []
        for r in results:
            cfg = configs.get(r.experiment_name)
            out.append(
                {
                    "name": r.experiment_name,
                    "description": cfg.description if cfg else "",
                    "segment": cfg.segment if cfg else "",
                    "control_description": cfg.control_description if cfg else "",
                    "treatment_description": cfg.treatment_description if cfg else "",
                    "control_conversions": r.control_conversions,
                    "control_trials": r.control_trials,
                    "treatment_conversions": r.treatment_conversions,
                    "treatment_trials": r.treatment_trials,
                    "prob_treatment_beats_control": r.prob_treatment_beats_control,
                    "expected_lift": r.expected_lift,
                    "recommended_sample_size": r.recommended_sample_size,
                    "posterior_data": r.posterior_data,
                    "created_at": r.created_at,
                }
            )
        return out
    finally:
        db.close()
