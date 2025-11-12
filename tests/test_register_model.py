import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import json
import joblib
from unittest.mock import patch, MagicMock

from mlops.modeling.register_model import ModelLoader, MetricsLoader, MLflowRegistrar


class DummyModel:
    def __init__(self):
        self.name = "dummy"


@pytest.fixture
def mock_model(tmp_path):
    """Crea un modelo simulado y lo guarda como joblib."""
    model = DummyModel()
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    return model, model_path


@pytest.fixture
def mock_metrics_file(tmp_path):
    """Crea un archivo JSON de métricas temporal."""
    metrics = {"accuracy": 0.95, "f1_macro": 0.90}
    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    return metrics, metrics_path


def test_model_loader_loads_model(mock_model):
    """Verifica que ModelLoader carga correctamente el modelo desde disco."""
    _, model_path = mock_model
    with patch("builtins.print"):
        model = ModelLoader.load(model_path)
    assert hasattr(model, "name")
    assert model.name == "dummy"


def test_metrics_loader_loads_metrics(mock_metrics_file):
    """Verifica que MetricsLoader lee correctamente el archivo JSON de métricas."""
    expected_metrics, metrics_path = mock_metrics_file
    with patch("builtins.print"):
        metrics = MetricsLoader.load(metrics_path)
    assert metrics == expected_metrics
    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)


@patch("mlflow.register_model")
@patch("mlflow.active_run")
@patch("mlflow.start_run")
@patch("mlflow.sklearn.log_model")
@patch("mlflow.log_metrics")
@patch("mlflow.set_experiment")
@patch("mlflow.set_tracking_uri")
def test_mlflow_registrar_registers_model(
    mock_set_tracking,
    mock_set_experiment,
    mock_log_metrics,
    mock_log_model,
    mock_start_run,
    mock_active_run,
    mock_register_model,
    mock_model,
    mock_metrics_file,
):
    """Verifica que MLflowRegistrar configura y registra el modelo correctamente."""
    model, _ = mock_model
    metrics, _ = mock_metrics_file

    # Configurar mocks
    mock_run_context = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_run_context
    mock_active_run.return_value.info.run_id = "1234abcd"

    registrar = MLflowRegistrar(
        tracking_uri="file:./mlruns_test", experiment_name="Test_Experiment"
    )

    with patch("builtins.print"):
        registrar.register_model(model, metrics, model_name="TestModel")

    # Verificar configuración
    mock_set_tracking.assert_called_once_with("file:./mlruns_test")
    mock_set_experiment.assert_called_once_with("Test_Experiment")

    # Verificar logging
    mock_log_metrics.assert_called_once_with(metrics)
    mock_log_model.assert_called_once_with(model, artifact_path="model")

    # Verificar registro en el registry
    mock_register_model.assert_called_once()
    args, kwargs = mock_register_model.call_args
    assert "runs:/1234abcd/model" in kwargs["model_uri"]
    assert kwargs["name"] == "TestModel"


def test_mlflow_registrar_configures_on_init():
    """Verifica que el constructor de MLflowRegistrar llama a la configuración de MLflow."""
    with patch("mlflow.set_tracking_uri") as mock_uri, patch("mlflow.set_experiment") as mock_exp:
        MLflowRegistrar(tracking_uri="file:./mlruns_ci", experiment_name="CI_Experiment")
        mock_uri.assert_called_once_with("file:./mlruns_ci")
        mock_exp.assert_called_once_with("CI_Experiment")
