.PHONY: setup data train lint test test-coverage

setup:
	pip install -r requirements.txt

data:
	dvc pull

train:
	python -m mlops.modeling.train

lint:
	python -m pip install pre-commit black ruff isort
	pre-commit install
	pre-commit run --all-files

test:
	pytest -v

test-coverage:
	pytest --cov=mlops --cov-report=term-missing -v

test-coverage-html:
	pytest --cov=mlops --cov-report=term-missing --cov-report=html -v
