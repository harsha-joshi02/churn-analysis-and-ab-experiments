.PHONY: help setup data db train experiment app test clean all

PYTHON    := python
PYTEST    := pytest
STREAMLIT := streamlit

help:
	@echo ""
	@echo "Churn Intelligence Platform — available commands"
	@echo "================================================"
	@echo "  make setup      Install Python dependencies"
	@echo "  make data       Download IBM Telco dataset to data/raw/"
	@echo "  make docker     Start PostgreSQL + MLflow containers"
	@echo "  make db         Initialise PostgreSQL schema"
	@echo "  make train      Train XGBoost (Optuna 20 trials) + MLflow"
	@echo "  make experiment Run sample Bayesian A/B experiment"
	@echo "  make app        Launch Streamlit dashboard → http://localhost:8501"
	@echo "  make test       Run pytest suite"
	@echo "  make clean      Remove generated artefacts"
	@echo "  make all        data → docker → db → train → experiment"
	@echo ""

setup:
	pip install -r requirements.txt

data:
	@mkdir -p data/raw
	curl -sL "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
		-o data/raw/telco_churn.csv
	@echo "Dataset downloaded → data/raw/telco_churn.csv"

docker:
	docker compose up -d
	@echo "Waiting for PostgreSQL to be healthy..."
	@until docker exec churn_postgres pg_isready -U churn -d churn_db > /dev/null 2>&1; do sleep 2; done
	@echo "PostgreSQL ready."

db:
	$(PYTHON) -c "from src.db import init_db; init_db(); print('DB schema ready.')"

train:
	$(PYTHON) -m src.train --n-trials 20

experiment:
	$(PYTHON) -c "\
from src.predict import load_customer_scores; \
from src.experiments import simulate_discount_experiment, save_experiment_to_db; \
from src.db import init_db; \
preds = load_customer_scores(); \
high = preds[preds['risk_segment']=='High']; \
exp, result = simulate_discount_experiment(high); \
init_db(); \
save_experiment_to_db(exp, result); \
print(f'P(treatment>control)={result.prob_treatment_beats_control:.1%}  lift={result.expected_lift:.1%}') \
"

app:
	$(STREAMLIT) run app/dashboard.py

test:
	$(PYTEST) tests/ -v --tb=short

clean:
	rm -rf data/processed/ models/ mlruns/

all: data docker db train experiment
	@echo "Pipeline complete — run 'make app' to launch the dashboard."
