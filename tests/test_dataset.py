import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
import yaml
from mlops.dataset import (
    ConfigLoader,
    OutlierFilter,
    DataCleaner,
    DataSplitter,
    DatasetProcessor,
)


# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture
def sample_df():
    """DataFrame pequeño de ejemplo."""
    return pd.DataFrame({
        "a": [1, 2, 3, 100],
        "b": [10, 20, 30, 40],
        "c": ["1", "2", "x", "3"],
        "target": [0, 1, 0, 1]
    })


@pytest.fixture
def sample_config(tmp_path):
    """Crea un archivo params.yaml temporal."""
    config_data = {
        "clean": {
            "numeric_cols": {"a": {"impute": "mean"}},
            "categorical_cols": {},
            "boolean_cols": [],
            "target_col": "target",
            "outliers": {"method": "iqr"}
        },
        "split": {
            "test_size": 0.25,
            "random_state": 123,
            "stratify": True
        }
    }
    path = tmp_path / "params.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return str(path)


# -------------------------------
# ConfigLoader
# -------------------------------
def test_config_loader_reads_yaml(sample_config):
    loader = ConfigLoader(sample_config)
    params = loader.load()
    assert "clean" in params
    assert "split" in params


# -------------------------------
# OutlierFilter
# -------------------------------
def test_iqr_filter_removes_outliers(sample_df):
    filtered = OutlierFilter.iqr_filter(sample_df, k=1.5)
    assert len(filtered) < len(sample_df)
    assert 100 not in filtered["a"].values


def test_zscore_filter_removes_outliers(sample_df):
    # usamos un z más bajo para forzar la detección
    filtered = OutlierFilter.zscore_filter(sample_df, z=1.0)
    assert len(filtered) < len(sample_df)
    assert 100 not in filtered["a"].values



# -------------------------------
# DataCleaner
# -------------------------------
def test_data_cleaner_creates_parquet(tmp_path, sample_config):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.parquet"

    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": [5, np.nan, 7, 8],
        "target": [10, 20, 30, 40]
    })
    df.to_csv(input_file, index=False)

    config = ConfigLoader(sample_config).load()
    cleaner = DataCleaner(config)
    cleaner.clean(input_file, output_file)

    assert output_file.exists()
    cleaned = pd.read_parquet(output_file)
    assert "a" in cleaned.columns
    assert not cleaned["a"].isna().any()


# -------------------------------
# DataSplitter
# -------------------------------
def test_data_splitter_creates_train_and_test(tmp_path, sample_config):
    input_file = tmp_path / "input.parquet"
    train_file = tmp_path / "train.parquet"
    test_file = tmp_path / "test.parquet"

    df = pd.DataFrame({
        "x1": np.random.rand(20),
        "x2": np.random.rand(20),
        "target": np.random.choice([0, 1], size=20)
    })
    df.to_parquet(input_file, index=False)

    config = ConfigLoader(sample_config).load()
    splitter = DataSplitter(config)
    splitter.split(input_file, train_file, test_file)

    assert train_file.exists()
    assert test_file.exists()

    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)
    assert "target" in train_df.columns
    assert len(train_df) > 0
    assert len(test_df) > 0


# -------------------------------
# DatasetProcessor
# -------------------------------
def test_dataset_processor_runs_clean_and_split(tmp_path, sample_config):
    csv_file = tmp_path / "data.csv"
    parquet_file = tmp_path / "cleaned.parquet"
    train_file = tmp_path / "train.parquet"
    test_file = tmp_path / "test.parquet"

    # Dataset más grande para evitar el error de estratificación
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8],
        "b": [5, 6, 7, 8, 9, 10, 11, 12],
        "target": [0, 1, 0, 1, 0, 1, 0, 1]
    })
    df.to_csv(csv_file, index=False)

    processor = DatasetProcessor(sample_config)
    processor.run_clean(csv_file, parquet_file)
    processor.run_split(parquet_file, train_file, test_file)

    assert parquet_file.exists()
    assert train_file.exists()
    assert test_file.exists()

    cleaned = pd.read_parquet(parquet_file)
    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    assert "target" in cleaned.columns
    assert len(train_df) > 0
    assert len(test_df) > 0

