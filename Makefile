.PHONY: help install install-dev lint format test test-smoke clean

PYTHON ?= python

help:
	@echo "PneumoShift Makefile commands:"
	@echo "  install        Install package dependencies"
	@echo "  install-dev    Install package with developer & test dependencies"
	@echo "  lint           Run flake8 and mypy checks"
	@echo "  format         Format code with black and isort"
	@echo "  test           Run all unit tests"
	@echo "  test-smoke     Run end-to-end smoke test with synthetic data"
	@echo "  clean          Clean build and cache artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,xai-extra]"

lint:
	flake8 src scripts tests
	mypy src

format:
	black src scripts tests configs
	isort src scripts tests

test:
	pytest tests/

test-smoke:
	pytest tests/test_smoke.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +

