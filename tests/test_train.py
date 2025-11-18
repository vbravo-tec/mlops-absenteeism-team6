import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import yaml

from mlops.modeling.train import ParamLoader, DataLoader, ModelFactory


@pytest.fixture
def sample_yaml(tmp_path):
    """Crea un archivo YAML temporal con configuración de modelo."""
    config = {
        "model": {"alg": "random_forest", "params": {"n_estimators": 10, "max_depth": 2}},
        "mlflow": {"experiment": "Test_Experiment"},
    }
    yaml_path = tmp_path / "params.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return yaml_path


@pytest.fixture
def sample_data(tmp_path):
    """Crea archivos Parquet simulados de entrenamiento y prueba."""
    df_train = pd.DataFrame(
        {"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 2, 3, 4], "target": [0, 1, 0, 1]}
    )
    df_test = pd.DataFrame({"f1": [0.5, 0.6], "f2": [5, 6], "target": [1, 0]})
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    return train_path, test_path


def test_param_loader_reads_yaml(sample_yaml):
    """Verifica que ParamLoader carga correctamente los parámetros."""
    loader = ParamLoader(path=sample_yaml)
    params = loader.get("model")
    assert "alg" in params
    assert params["alg"] == "random_forest"


def test_data_loader_splits_xy(sample_data):
    """Verifica que DataLoader separa correctamente X e y."""
    train_path, _ = sample_data
    X, y = DataLoader.load_xy(train_path)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "target" not in X.columns
    assert len(X) == len(y)


def test_model_factory_creates_correct_model():
    """Verifica que ModelFactory crea el modelo adecuado según el algoritmo."""
    rf = ModelFactory.create("random_forest", {"n_estimators": 5})
    lr = ModelFactory.create("logreg", {"max_iter": 10})
    assert rf.__class__.__name__ == "RandomForestClassifier"
    assert lr.__class__.__name__ == "LogisticRegression"

    # Modelo no soportado
    with pytest.raises(ValueError):
        ModelFactory.create("unsupported_model", {})
