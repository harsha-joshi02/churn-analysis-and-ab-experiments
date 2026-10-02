# Churn Intelligence Platform

**SaaS churn prediction and experimentation system**

---

## Problem Statement

Subscription businesses lose 20–30% of their customer base annually to churn. Identifying *who* will churn, *why* they are at risk, and *which interventions* are effective are three distinct data science problems. This platform solves all three:

1. **Predict** — XGBoost classifier ranks every customer by churn probability
2. **Explain** — SHAP values expose the top drivers for each individual
3. **Experiment** — Bayesian A/B engine measures whether interventions (e.g. discounts) improve retention

---

## Dataset

**IBM Telco Customer Churn** — 7,043 customer records, 19 predictor columns, 26.5% churn rate.

Source: [IBM Sample Data Sets](https://github.com/IBM/telco-customer-churn-on-icp4d)

Downloaded automatically via `make data`.

---

## Architecture

```
churn_platform/
├── data/
│   └── raw/telco_churn.csv          # IBM Telco dataset (downloaded via make data)
├── src/
│   ├── features.py                  # Feature engineering + encoding
│   ├── train.py                     # XGBoost + Optuna HPO + MLflow
│   ├── predict.py                   # Inference + risk segmentation
│   ├── experiments.py               # Bayesian A/B engine (Beta conjugate)
│   ├── db.py                        # SQLAlchemy models + PostgreSQL
│   └── utils.py                     # Logging, paths, config
├── app/
│   └── dashboard.py                 # Streamlit 4-page dashboard
├── notebooks/
│   └── eda.ipynb                    # Exploratory data analysis
├── tests/
│   └── test_experiments.py          # Unit tests for Bayesian engine
├── mlruns/                          # Portable MLflow backend + artifacts (git-ignored)
├── models/                          # Trained model + plots (git-ignored)
├── docker-compose.yml               # PostgreSQL + MLflow services
├── Makefile                         # Task automation
└── requirements.txt
```

---

## ML Pipeline

### Feature Engineering (`src/features.py`)

| Feature | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months as a customer |
| `monthly_charges` | Numeric | Current monthly bill |
| `total_charges` | Numeric | Lifetime spend |
| `contract_type` | Encoded | Month-to-month / One year / Two year |
| `payment_method` | Encoded | Electronic check / Bank transfer / etc. |
| `internet_service` | Encoded | DSL / Fiber optic / No |
| `senior_citizen` | Binary | Senior customer flag |
| `partner` / `dependents` | Binary | Demographic signals |
| `phone_service` + 7 add-ons | Binary | Active service subscriptions |
| **`num_products`** | Derived | Count of active add-on services |
| **`charge_per_product`** | Derived | `monthly_charges / (num_products + 1)` |
| **`tenure_x_charge`** | Derived | Interaction term (sticky high-value customers) |
| **`avg_monthly_over_total`** | Derived | Monthly charge relative to lifetime spend |

### Model Training (`src/train.py`)

```
XGBoostClassifier
  └── Optuna TPE sampler (20 trials)
        └── Maximise AUC-ROC on clean validation set
              └── scale_pos_weight tuned around real class ratio (no SMOTE)
                    └── MLflow: params, metrics, plots, model artefact
```

**Note on methodology:** SMOTE is not used. Optuna tunes `scale_pos_weight` around the training-set class ratio to handle class imbalance while keeping validation and test data unchanged.

### Evaluation (on clean 20% holdout)

| Metric | Value |
|---|---|
| AUC-ROC | ~0.847 |
| F1 Score | ~0.602 |
| Avg Precision | ~0.661 |

Training writes two distinct prediction outputs:

- `models/holdout_predictions.parquet` contains only untouched test rows and is used for evaluation metrics.
- `models/customer_scores.parquet` contains scores for the full customer population and is used by operational dashboard views.

### Risk Segmentation

| Segment | Churn Probability |
|---|---|
| Low | < 30% |
| Medium | 30%–60% |
| High | > 60% |

---

## Experimentation Platform

### Bayesian A/B Testing (`src/experiments.py`)

```
Retention modelled as Bernoulli(θ)
  Prior:     θ ~ Beta(α₀, β₀)   [uninformative: α₀=β₀=1]
  Likelihood: k successes in n trials
  Posterior: θ | data ~ Beta(α₀+k, β₀+n-k)

P(treatment > control) = Monte-Carlo integral over posterior samples
Expected lift          = (treatment posterior mean - control posterior mean)
                         / control posterior mean
```

### Why Bayesian over Frequentist

| | Frequentist | Bayesian |
|---|---|---|
| **Output** | p < 0.05 → reject H₀ | P(treatment > control) = 97.9% |
| **Peeking** | Invalidates the test | Posterior updates continuously |
| **Decision** | Binary (significant/not) | Probabilistic, quantifies uncertainty |

---

## Dashboard (`app/dashboard.py`)

| Page | What it shows |
|---|---|
| **Churn Overview** | KPI strip, risk segment donut, churn histogram, tenure by segment, churn by contract type |
| **Model Performance** | AUC-ROC, F1, Avg Precision + interactive ROC/PR curves, feature importance, SHAP beeswarm |
| **Customer Explorer** | Filterable customer table + per-customer churn gauge + SHAP waterfall |
| **Experiments** | Bayesian A/B simulator + posterior distribution chart + experiment history (PostgreSQL) |

---

## Quick Start

### Prerequisites
- Python ≥ 3.11
- Docker Desktop

```bash
# 1. Clone the repo
git clone <repo-url> && cd churn_platform

# 2. Install dependencies
make setup

# 3. Download IBM Telco dataset
make data

# 4. Start PostgreSQL + MLflow
make docker

# 5. Initialise DB schema
make db

# 6. Train model (~2 min)
make train

# 7. Launch dashboard
make app
# → http://localhost:8501
```

Or run everything in one command:
```bash
make all && make app
```

---

## Environment Variables

Copy `.env.example` to `.env` (defaults work out of the box with Docker):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://churn:churnpass@localhost:5432/churn_db` | PostgreSQL connection |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | MLflow tracking server |
| `MLFLOW_EXPERIMENT_NAME` | `churn_prediction` | MLflow experiment name |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | pandas, numpy, pyarrow |
| ML | XGBoost, scikit-learn |
| HPO | Optuna (TPE sampler) |
| Explainability | SHAP |
| Statistics | scipy (Beta distribution) |
| Experiment Tracking | MLflow |
| Database | PostgreSQL + SQLAlchemy 2.0 |
| Dashboard | Streamlit + Plotly |
| Infrastructure | Docker Compose |
| Testing | pytest |
