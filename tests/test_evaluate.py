import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import json
import joblib
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlops.modeling.evaluate import (
    ModelLoader,
    TestDataLoader,
    ModelEvaluator
)

class DummyModel:
    def predict(self, X=None):
        import numpy as np
        return np.zeros(len(X), dtype=int)  # 0 para todos

    def predict_proba(self, X=None):
        import numpy as np
        n = len(X)
        return np.tile([[0.8, 0.2]], (n, 1))  # misma probabilidad para todas



@pytest.fixture
def sample_df(tmp_path):
    """Crea un DataFrame simulado y lo guarda como parquet."""
    df = pd.DataFrame({
        "feature1": [0.2, 0.4, 0.6, 0.8],
        "feature2": [1, 0, 1, 0],
        "target": [0, 1, 0, 1]
    })
    parquet_path = tmp_path / "test.parquet"
    df.to_parquet(parquet_path, index=False)
    return df, parquet_path

@pytest.fixture
def mock_model(tmp_path):
    model = DummyModel()
    model_path = tmp_path / "model.joblib"
    import joblib
    joblib.dump(model, model_path)
    return model, model_path


def test_model_loader_loads_model(mock_model):
    """Verifica que el modelo se carga correctamente desde disco."""
    _, model_path = mock_model
    with patch("builtins.print"):
        loaded_model = ModelLoader.load(model_path)
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")


def test_testdataloader_loads_data(sample_df):
    """Verifica que TestDataLoader carga correctamente el dataset."""
    df, parquet_path = sample_df
    with patch("builtins.print"):
        X, y = TestDataLoader.load(parquet_path)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "target" not in X.columns
    assert len(X) == len(y)


def test_testdataloader_raises_if_no_target(tmp_path):
    """Verifica que falle si falta la columna target."""
    df = pd.DataFrame({"feature1": [1, 2, 3]})
    parquet_path = tmp_path / "bad.parquet"
    df.to_parquet(parquet_path, index=False)

    with pytest.raises(AssertionError):
        TestDataLoader.load(parquet_path)


def test_modelevaluator_evaluate_generates_results(sample_df, mock_model):
    """Verifica que se calculan métricas correctamente."""
    df, _ = sample_df
    model, _ = mock_model
    X = df.drop(columns=["target"])
    y = df["target"]

    evaluator = ModelEvaluator(model, X, y)
    results = evaluator.evaluate()

    assert "classification_report" in results
    assert "confusion_matrix" in results
    assert isinstance(results["classification_report"], dict)
    assert isinstance(results["confusion_matrix"], list)
    # ROC AUC puede ser None o float dependiendo del modelo
    assert results["roc_auc"] is None or isinstance(results["roc_auc"], float)


def test_modelevaluator_save_report(tmp_path, sample_df, mock_model):
    """Verifica que el reporte se guarda correctamente como JSON."""
    df, _ = sample_df
    model, _ = mock_model
    X = df.drop(columns=["target"])
    y = df["target"]

    evaluator = ModelEvaluator(model, X, y)
    evaluator.evaluate()

    report_path = tmp_path / "report.json"
    with patch("builtins.print"):
        evaluator.save_report(report_path)

    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "classification_report" in data
    assert "confusion_matrix" in data


def test_modelevaluator_handles_predict_proba_errors(sample_df):
    """Verifica que si el modelo no tiene predict_proba no falle."""
    class ModelNoProba:
        def predict(self, X):
            return np.zeros(len(X))

    df, _ = sample_df
    X = df.drop(columns=["target"])
    y = df["target"]
    model = ModelNoProba()

    evaluator = ModelEvaluator(model, X, y)
    results = evaluator.evaluate()

    assert "roc_auc" in results
    assert results["roc_auc"] is None
