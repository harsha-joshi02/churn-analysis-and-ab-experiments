"""
Generate a synthetic SaaS / telco-style churn dataset (~7 000 rows).

Feature correlations are deliberately designed to produce a realistic churn
rate (~26 %) with the following ground-truth drivers:
  - Month-to-month contract  → highest churn risk
  - Electronic check payment → moderate risk signal
  - High support tickets     → churn signal
  - Short tenure             → churn signal
  - High monthly charges     → moderate risk
  - Low last-login recency   → churn signal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


def generate_churn_dataset(n_samples: int = 7043, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Categorical features ──────────────────────────────────────────────────
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.24, 0.21],
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
        p=[0.34, 0.22, 0.22, 0.22],
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_samples,
        p=[0.34, 0.44, 0.22],
    )

    # ── Numeric features (correlated with contract type) ──────────────────────
    is_monthly = (contract_type == "Month-to-month").astype(float)
    is_annual = (contract_type == "One year").astype(float)

    tenure = np.where(
        contract_type == "Two year",
        rng.integers(18, 73, n_samples),
        np.where(
            contract_type == "One year",
            rng.integers(6, 49, n_samples),
            rng.integers(0, 37, n_samples),
        ),
    ).astype(int)

    base_charge = np.where(
        internet_service == "Fiber optic",
        rng.uniform(65, 115, n_samples),
        np.where(
            internet_service == "DSL",
            rng.uniform(45, 82, n_samples),
            rng.uniform(20, 52, n_samples),
        ),
    )
    monthly_charges = base_charge.round(2)

    num_products = rng.integers(1, 7, n_samples)

    # Support tickets: monthly customers open more tickets on average
    support_base = 1.5 + 1.5 * is_monthly + 0.5 * is_annual
    support_tickets = rng.poisson(support_base).astype(int)

    # Login recency: churners haven't logged in recently
    last_login_days_ago = rng.integers(0, 61, n_samples)

    # ── Churn probability via logistic model ──────────────────────────────────
    # Intercept calibrated so that avg churn ≈ 26%
    logit = (
        -2.8                                                   # intercept
        + 1.2  * is_monthly                                    # contract risk
        + 0.4  * (contract_type == "One year").astype(float)
        - 0.02 * tenure                                        # loyalty
        + 0.008 * monthly_charges                              # price sensitivity
        - 0.05 * num_products                                  # product stickiness
        + 0.12 * support_tickets                               # frustration
        + 0.015 * last_login_days_ago                          # disengagement
        + 0.30 * (payment_method == "Electronic check").astype(float)
        + 0.15 * (internet_service == "Fiber optic").astype(float)
        + rng.normal(0, 0.5, n_samples)                        # noise
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n_samples) < churn_prob).astype(int)

    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_samples + 1)]

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "num_products": num_products,
            "support_tickets": support_tickets,
            "last_login_days_ago": last_login_days_ago,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "churn": churn,
        }
    )

    return df


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn dataset")
    parser.add_argument("--n-samples", type=int, default=7043)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=RAW_DIR / "telco_churn.csv")
    opts = parser.parse_args(args)

    opts.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_churn_dataset(n_samples=opts.n_samples, seed=opts.seed)
    df.to_csv(opts.output, index=False)

    churn_rate = df["churn"].mean()
    print(f"Generated {len(df):,} rows  →  {opts.output}")
    print(f"Churn rate : {churn_rate:.1%}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
