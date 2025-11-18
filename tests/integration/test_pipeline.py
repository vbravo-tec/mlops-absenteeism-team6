import subprocess
import pandas as pd
import pytest


@pytest.mark.integration
def test_clean_stage(tmp_path):
    """Ejecuta la etapa de limpieza y valida la salida intermedia."""
    cmd = [
        "python",
        "-m",
        "mlops.dataset",
        "--task",
        "clean",
        "--input",
        "tests/integration/test_data/work_absenteeism_modified.csv",
        "--output",
        tmp_path / "clean.parquet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Falló clean: {result.stderr}"

    df = pd.read_parquet(tmp_path / "clean.parquet")
    assert not df.empty
    assert df.isnull().sum().sum() < len(df), "Demasiados nulos tras limpiar"


@pytest.mark.integration
def test_train_stage(tmp_path):
    """Ejecuta la etapa de entrenamiento y verifica métricas."""
    train_path = "tests/integration/test_data/train.parquet"
    test_path = "tests/integration/test_data/test.parquet"
    model_out = tmp_path / "model.pkl"
    metrics_out = tmp_path / "metrics.json"

    result = subprocess.run(
        [
            "python",
            "-m",
            "mlops.modeling.train",
            "--train",
            train_path,
            "--test",
            test_path,
            "--model-out",
            model_out,
            "--metrics-out",
            metrics_out,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Entrenamiento falló: {result.stderr}"
    assert model_out.exists()
    assert metrics_out.exists()
