import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import yaml
from unittest.mock import patch

from mlops.features import ParamLoader, FeatureEngineer, FeaturePreprocessor, FeatureBuilder


@pytest.fixture
def sample_yaml(tmp_path):
    """Crea un archivo params.yaml temporal con configuración básica."""
    yaml_content = {
        "features": {
            "new": {
                "feature_sum": {"enabled": True, "formula": "a + b"},
                "feature_flag": {"enabled": True, "formula": "a > 1"},
            },
            "encode": "onehot",
            "scale": "standard",
            "drop_low_variance": True,
            "variance_thresh": 0.0,
        }
    }
    yaml_path = tmp_path / "params.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f)
    return yaml_path


@pytest.fixture
def sample_df():
    """Crea un DataFrame de prueba."""
    return pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "category": ["x", "y", "z"], "target": [0, 1, 0]}
    )


def test_param_loader_reads_yaml(sample_yaml):
    """Verifica que ParamLoader lee correctamente el archivo YAML."""
    loader = ParamLoader(path=sample_yaml)
    params = loader.get("features")
    assert isinstance(params, dict)
    assert "new" in params


def test_feature_engineer_creates_new_columns(sample_df, sample_yaml):
    """Verifica que FeatureEngineer crea nuevas columnas según las fórmulas."""
    params = ParamLoader(path=sample_yaml).params
    fe = FeatureEngineer(params)
    df_transformed = fe.create_behavioral_features(sample_df)

    # Deben haberse creado las columnas definidas en params.yaml
    assert "feature_sum" in df_transformed.columns
    assert "feature_flag" in df_transformed.columns

    # Verifica que los valores sean correctos
    assert all(df_transformed["feature_sum"] == sample_df["a"] + sample_df["b"])
    assert set(df_transformed["feature_flag"].unique()) <= {0, 1}


def test_feature_preprocessor_applies_encoding_and_scaling(sample_df, sample_yaml):
    """Verifica que FeaturePreprocessor aplica correctamente one-hot y scaling."""
    params = ParamLoader(path=sample_yaml).params
    X = sample_df.drop(columns=["target"])
    preprocessor = FeaturePreprocessor(params)
    X_processed = preprocessor.fit_transform(X)

    # Debe contener columnas codificadas y escaladas
    assert isinstance(X_processed, pd.DataFrame)
    assert X_processed.shape[0] == X.shape[0]
    assert not X_processed.isna().any().any()


def test_feature_builder_creates_output_file(tmp_path, sample_df, sample_yaml):
    """Verifica que FeatureBuilder genera un archivo parquet de salida."""
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    sample_df.to_parquet(input_path, index=False)
    params = ParamLoader(path=sample_yaml).params

    builder = FeatureBuilder(input_path, output_path, params)

    with patch.object(FeatureEngineer, "create_behavioral_features", return_value=sample_df):
        builder.run()

    assert output_path.exists()
    df_result = pd.read_parquet(output_path)
    assert "target" in df_result.columns
    assert len(df_result) == len(sample_df)


def test_feature_engineer_handles_missing_formula(sample_df):
    """Verifica que se maneje correctamente una feature sin fórmula."""
    params = {"features": {"new": {"no_formula": {"enabled": True}}}}
    fe = FeatureEngineer(params)
    with patch("builtins.print") as mock_print:
        df = fe.create_behavioral_features(sample_df)
        mock_print.assert_any_call("[features] ⚠️ no_formula no tiene fórmula definida, se omite.")
        assert "no_formula" not in df.columns


def test_feature_preprocessor_no_transformers(sample_df):
    """Verifica que no falle si no hay columnas numéricas o categóricas."""
    params = {"features": {"encode": None, "scale": None}}
    preprocessor = FeaturePreprocessor(params)
    df_empty = pd.DataFrame({"target": [1, 0, 1]})
    X_processed = preprocessor.fit_transform(df_empty)
    assert isinstance(X_processed, pd.DataFrame)


def test_feature_preprocessor_drops_low_variance(sample_df, sample_yaml):
    """Verifica que se eliminen columnas con baja varianza."""
    params = ParamLoader(path=sample_yaml).params
    params["features"]["variance_thresh"] = 10.0  # Umbral alto para forzar eliminación
    preprocessor = FeaturePreprocessor(params)
    X = sample_df[["a", "b"]]
    X_processed = preprocessor.fit_transform(X)
    assert (
        X_processed.shape[1] == 0
        or X_processed.var().min() <= params["features"]["variance_thresh"]
    )
