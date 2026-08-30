.PHONY: install test lint train train-xgboost train-rf train-demo features demo-data api dashboard docker-up docker-down help

# ── Help ────────────────────────────────────────────────────────────────────
help:
	@echo "Real Estate Price Prediction — common commands"
	@echo ""
	@echo "  make install     Install all dependencies"
	@echo "  make test        Run unit tests (no network)"
	@echo "  make lint        Run ruff linter"
	@echo "  make train       Train linear_regression model on engineered data (current best per reports/model_comparison.csv; fails if no real data)"
	@echo "  make train-xgboost  Train XGBoost instead (for comparison — NOT the production default, see MODEL_CARD.md)"
	@echo "  make demo-data   Explicitly generate SYNTHETIC demo data (data/processed/*.synthetic.csv)"
	@echo "  make train-demo  Explicitly train on SYNTHETIC demo data (tags artefact _synthetic_)"
	@echo "  make api         Start FastAPI server on port 8000"
	@echo "  make dashboard   Start Streamlit dashboard on port 8501"
	@echo "  make docker-up   Launch API + dashboard with Docker Compose"
	@echo "  make docker-down Stop Docker services"

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# ── Testing ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

# ── Linting ──────────────────────────────────────────────────────────────────
lint:
	ruff check src/ api/ dashboard/ scripts/ tests/ --fix

# ── ML Pipeline ──────────────────────────────────────────────────────────────
# `features`/`train` are fail-closed: they error out if no real data exists,
# instead of silently generating synthetic data. Use the `demo-data`/
# `train-demo` targets to explicitly opt into synthetic/demo data.
features:
	python scripts/run_feature_engineering_real.py

demo-data:
	python scripts/run_feature_engineering_real.py --allow-synthetic

train:
	python -X utf8 scripts/run_model_training_real.py --model linear_regression

train-xgboost:
	python -X utf8 scripts/run_model_training_real.py --model xgboost

train-rf:
	python -X utf8 scripts/run_model_training_real.py --model random_forest

train-demo:
	python -X utf8 scripts/run_model_training_real.py --model linear_regression --allow-synthetic

# ── Services ─────────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

# ── Docker ───────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
