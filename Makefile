.PHONY: clean lint format test update_schema

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
NAME = gnn_180b
SRC_DIR = gnn_180b

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
	poetry run isort --skip-glob $(SRC_DIR)/tests --skip-glob $(SRC_DIR)/notebooks --profile=black --lines-after-imports 2 --check-only $(SRC_DIR) main.py
	poetry run black --check $(SRC_DIR) main.py --diff
	poetry run flake8 --exclude $(SRC_DIR)/tests --exclude $(SRC_DIR)/notebooks --ignore=W503,E501 $(SRC_DIR) main.py

format:
	@echo ">>> Reformatting code"
	poetry run isort --skip-glob $(SRC_DIR)/tests --skip-glob $(SRC_DIR)/notebooks --profile=black --lines-after-imports 2 $(SRC_DIR) main.py
	poetry run black $(SRC_DIR) main.py

test:
	@echo ">>> Running tests"
	poetry run pytest