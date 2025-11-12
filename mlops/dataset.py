import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore


class ConfigLoader:
    """Carga la configuración desde un archivo YAML."""

    def __init__(self, path="params.yaml"):
        self.path = path

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


class OutlierFilter:
    """Provee métodos para eliminar outliers usando diferentes estrategias."""

    @staticmethod
    def iqr_filter(df, k=1.5, numeric_cols=None):
        numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns
        mask = pd.Series(True, index=df.index)
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - k * IQR, Q3 + k * IQR
            mask &= df[col].between(lower, upper) | df[col].isna()
        return df[mask]

    @staticmethod
    def zscore_filter(df, z=3.0, numeric_cols=None):
        numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns
        zscores = df[numeric_cols].apply(lambda s: np.abs(zscore(s, nan_policy="omit")))
        mask = (zscores <= z) | zscores.isna()
        return df[mask.all(axis=1)]


class DataCleaner:
    """Limpia y transforma datasets según la configuración."""

    def __init__(self, config: dict):
        self.config = config.get("clean", {})
        self.outlier_filter = OutlierFilter()

    def _impute_column(self, df, col, meta):
        method = meta.get("impute", "none")
        if method == "mean":
            val = df[col].mean()
            df[col] = df[col].fillna(val)
        elif method == "median":
            val = df[col].median()
            df[col] = df[col].fillna(val)
        elif method == "most_frequent":
            val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
            df[col] = df[col].fillna(val)
        elif method == "fill_zero":
            df[col] = df[col].fillna(0)
        elif method == "none":
            df = df.dropna(subset=[col])
        else:
            raise ValueError(f"Método de imputación desconocido: {method}")
        return df

    def _validate_range(self, df, col, meta):
        if "range" in meta:
            min_val, max_val = meta["range"]
            df = df[df[col].between(min_val, max_val, inclusive="both")]
        return df

    def _clean_categorical(self, df):
        for col, meta in self.config.get("categorical_cols", {}).items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = self._validate_range(df, col, meta)
                if meta.get("impute") == "most_frequent":
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                    df[col] = df[col].fillna(mode_val)
                elif meta.get("impute") == "fill_zero":
                    df[col] = df[col].fillna(0)
                df[col] = df[col].astype(int).astype(str)
        return df

    def _clean_numeric(self, df):
        for col, meta in self.config.get("numeric_cols", {}).items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = self._validate_range(df, col, meta)
                df = self._impute_column(df, col, meta)
                df[col] = df[col].astype(float)
        return df

    def _clean_boolean(self, df):
        for col in self.config.get("boolean_cols", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df[df[col].isin([0, 1])]
                df[col] = df[col].astype(bool)
        return df

    def clean(self, input_path, output_path):
        df = pd.read_csv(input_path)
        drop_cols = self.config.get("drop_columns", [])
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        df = self._clean_categorical(df)
        df = self._clean_numeric(df)
        df = self._clean_boolean(df)

        target_col = self.config.get("target_col", "Absenteeism time in hours")
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df = df.dropna(subset=[target_col])
            df[target_col] = df[target_col].astype(float)

        # Outlier removal
        out_cfg = self.config.get("outliers", {})
        method = out_cfg.get("method", "iqr")
        if method == "iqr":
            df = self.outlier_filter.iqr_filter(df)
        elif method == "zscore":
            df = self.outlier_filter.zscore_filter(df, z=float(out_cfg.get("z_thresh", 3.0)))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"[clean] wrote {output_path} shape={df.shape}")


class DataSplitter:
    """Divide un dataset en conjuntos de entrenamiento y prueba."""

    def __init__(self, config: dict):
        self.config = config.get("split", {})

    def split(self, input_path, train_path, test_path):
        df = pd.read_parquet(input_path)
        test_size = float(self.config.get("test_size", 0.2))
        random_state = int(self.config.get("random_state", 42))
        stratify_flag = bool(self.config.get("stratify", True))

        target_col = "target" if "target" in df.columns else None
        if target_col is None:
            df["target"] = (
                df.select_dtypes(include=[np.number]).sum(axis=1)
                > df.select_dtypes(include=[np.number]).sum(axis=1).median()
            ).astype(int)
            target_col = "target"

        X, y = df.drop(columns=[target_col]), df[target_col]
        stratify = y if stratify_flag and y.nunique() > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        Path(train_path).parent.mkdir(parents=True, exist_ok=True)
        Path(test_path).parent.mkdir(parents=True, exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_parquet(train_path, index=False)
        pd.concat([X_test, y_test], axis=1).to_parquet(test_path, index=False)
        print(f"[split] wrote {train_path} {X_train.shape} and {test_path} {X_test.shape}")


class DatasetProcessor:
    """Punto de entrada principal, orquesta tareas de limpieza y división."""

    def __init__(self, config_path="params.yaml"):
        self.config = ConfigLoader(config_path).load()

    def run_clean(self, input_path, output_path):
        DataCleaner(self.config).clean(input_path, output_path)

    def run_split(self, input_path, train_path, test_path):
        DataSplitter(self.config).split(input_path, train_path, test_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["clean", "split"])
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--train")
    parser.add_argument("--test")
    args = parser.parse_args()

    processor = DatasetProcessor()
    if args.task == "clean":
        assert args.input and args.output, "--input/--output requeridos"
        processor.run_clean(args.input, args.output)
    elif args.task == "split":
        assert args.input and args.train and args.test, "--input/--train/--test requeridos"
        processor.run_split(args.input, args.train, args.test)


if __name__ == "__main__":
    main()
