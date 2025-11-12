import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


class ParamLoader:
    """Encargado de leer el archivo de parámetros YAML."""
    
    def __init__(self, path="params.yaml"):
        self.path = path
        self.params = self._load_params()
    
    def _load_params(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def get(self, key, default=None):
        return self.params.get(key, default)


class FeatureEngineer:
    """Crea nuevas columnas de comportamiento según las reglas del YAML."""

    def __init__(self, params):
        self.params = params

    def create_behavioral_features(self, df):
        df = df.copy()
        feat_cfg = self.params.get("features", {}).get("new", {})

        for name, meta in feat_cfg.items():
            if not meta.get("enabled", False):
                continue

            formula = meta.get("formula")
            output_col = meta.get("output_col", name)

            if not formula:
                print(f"[features] ⚠️ {name} no tiene fórmula definida, se omite.")
                continue

            try:
                df[output_col] = df.eval(formula, engine="python")

                if df[output_col].dtype == bool:
                    df[output_col] = df[output_col].astype(int)

                print(f"[features] ✅ '{output_col}' creado con fórmula: {formula}")

            except Exception as e:
                print(f"[features] ❌ Error al crear '{output_col}' → {e}")

        return df


class FeaturePreprocessor:
    """Encargado del encoding, escalado y reducción de varianza."""

    def __init__(self, params):
        fcfg = params.get("features", {})
        self.encode = fcfg.get("encode", "onehot")
        self.scale = fcfg.get("scale", "standard")
        self.drop_low_var = bool(fcfg.get("drop_low_variance", False))
        self.var_thresh = float(fcfg.get("variance_thresh", 0.0))
        self.column_transformer = None

    def fit_transform(self, X):
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        transformers = []

        if cat_cols and self.encode == "onehot":
            transformers.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

        if num_cols:
            if self.scale == "standard":
                transformers.append(("scale", StandardScaler(), num_cols))
            elif self.scale == "minmax":
                transformers.append(("scale", MinMaxScaler(), num_cols))

        if transformers:
            self.column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                verbose_feature_names_out=False
            )
            X_arr = self.column_transformer.fit_transform(X)

            try:
                out_cols = self.column_transformer.get_feature_names_out().tolist()
            except Exception:
                out_cols = [f"f{i}" for i in range(X_arr.shape[1])]

            X_feat = pd.DataFrame(X_arr, columns=out_cols)
        else:
            X_feat = X.copy()

        if self.drop_low_var and X_feat.shape[1] > 0:
            vars_ = X_feat.var(numeric_only=True)
            keep = vars_[vars_ > self.var_thresh].index.tolist()
            if keep:
                X_feat = X_feat[keep]

        return X_feat


class FeatureBuilder:
    """Pipeline principal: lectura, ingeniería, preprocesamiento y guardado."""

    def __init__(self, input_path, output_path, params):
        self.input_path = input_path
        self.output_path = output_path
        self.params = params
        self.engineer = FeatureEngineer(params)
        self.preprocessor = FeaturePreprocessor(params)

    def run(self):
        df = pd.read_parquet(self.input_path)
        df = self.engineer.create_behavioral_features(df)

        y = df["target"] if "target" in df.columns else None
        X = df.drop(columns=["target"]) if "target" in df.columns else df.copy()

        X_feat = self.preprocessor.fit_transform(X)

        if y is not None:
            X_feat["target"] = y.values

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        X_feat.to_parquet(self.output_path, index=False)
        print(f"[features] ✅ Archivo generado: {self.output_path} shape={X_feat.shape}")


class FeaturePipelineCLI:
    """Interfaz de línea de comandos para ejecutar el pipeline."""

    def __init__(self):
        self.args = self._parse_args()
        self.params = ParamLoader().params

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="Feature Builder CLI")
        parser.add_argument("--task", required=True, choices=["build"], help="Tarea a ejecutar")
        parser.add_argument("--input", required=True, help="Ruta del archivo de entrada")
        parser.add_argument("--output", required=True, help="Ruta del archivo de salida")
        return parser.parse_args()

    def run(self):
        if self.args.task == "build":
            builder = FeatureBuilder(self.args.input, self.args.output, self.params)
            builder.run()


def main():
    FeaturePipelineCLI().run()


if __name__ == "__main__":
    main()
