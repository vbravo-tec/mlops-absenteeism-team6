.PHONY: setup data train lint

setup:
\tpip install -r requirements.txt

data:
\tdvc pull

train:
\tpython -m mlops.modeling.train

lint:
\tpython -m pip install pre-commit black ruff isort
\tpre-commit install
\tpre-commit run --all-files
