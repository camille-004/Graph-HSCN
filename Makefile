.PHONY: create_environment install clean lint format baselines

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
NAME = graph_hscn
SRC_DIR = graph_hscn

# Set up Python interpreter environment
create_environment:
	@echo ">>> Creating an environment"
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true

# Install Python dependencies
install: create_environment
	@echo ">>> Installing Python dependencies"
	poetry install && poetry update

# Delete compiled Python files
clean:
	@echo ">>> Cleaning project"
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	@echo ">>> Running code linting"
	poetry run isort --skip-glob $(SRC_DIR)/tests --skip-glob $(SRC_DIR)/notebooks --profile=black --lines-after-imports 2 --check-only $(SRC_DIR) run.py
	poetry run black --check $(SRC_DIR) run.py agg_results/agg_runs.py --diff
	poetry run flake8 --exclude $(SRC_DIR)/tests --exclude $(SRC_DIR)/notebooks --ignore=W503,E501,F841,E203,D107,D403 $(SRC_DIR) run.py agg_results/agg_runs.py --docstring-convention numpy

format:
	@echo ">>> Reformatting code"
	poetry run isort --skip-glob $(SRC_DIR)/tests --skip-glob $(SRC_DIR)/notebooks --profile=black --lines-after-imports 2 $(SRC_DIR) run.py agg_results/agg_runs.py
	poetry run black $(SRC_DIR) run.py agg_results/agg_runs.py

baselines:
	@echo ">>> Running baseline MPNNs on resampled citation networks"
	sh run/run_baselines_citation.sh
