import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_behavioral_features(df, P):
    """

    Crea nuevas columnas de comportamiento basadas en expresiones definidas
    dinámicamente en params.yaml bajo 'features.new'.

    """
    df = df.copy()
    feat_cfg = P.get("features", {}).get("new", {})

    for name, meta in feat_cfg.items():
        if not meta.get("enabled", False):
            continue

        formula = meta.get("formula")
        output_col = meta.get("output_col", name)

        if not formula:
            print(f"[features] ⚠️ {name} no tiene fórmula definida, se omite.")
            continue

        try:
            # Evaluar fórmula de manera vectorizada
            df[output_col] = df.eval(formula, engine="python")

            # Si el resultado es booleano, convertir a 0/1
            if df[output_col].dtype == bool:
                df[output_col] = df[output_col].astype(int)

            print(f"[features] ✅ '{output_col}' creado con fórmula: {formula}")

        except Exception as e:
            print(f"[features] ❌ Error al crear '{output_col}' → {e}")

    return df


def build_features(input_path, output_path, P):
    df = pd.read_parquet(input_path)

    # Añadir nuevas features
    df = create_behavioral_features(df, P)

    y = df["target"] if "target" in df.columns else None
    X = df.drop(columns=["target"]) if "target" in df.columns else df.copy()

    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    fcfg = P.get("features", {})
    encode = fcfg.get("encode", "onehot")
    scale = fcfg.get("scale", "standard")
    drop_low_var = bool(fcfg.get("drop_low_variance", False))
    var_thresh = float(fcfg.get("variance_thresh", 0.0))

    transformers = []

    # Encoding
    if cat_cols and encode == "onehot":
        transformers.append(
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )

    # Scaling
    if num_cols:
        if scale == "standard":
            transformers.append(("scale", StandardScaler(), num_cols))
        elif scale == "minmax":
            transformers.append(("scale", MinMaxScaler(), num_cols))

    if transformers:
        ct = ColumnTransformer(
            transformers=transformers, remainder="drop", verbose_feature_names_out=False
        )
        X_arr = ct.fit_transform(X)
        try:
            out_cols = ct.get_feature_names_out().tolist()
        except Exception:
            out_cols = [f"f{i}" for i in range(X_arr.shape[1])]
        X_feat = pd.DataFrame(X_arr, columns=out_cols)
    else:
        X_feat = X.copy()

    if drop_low_var and X_feat.shape[1] > 0:
        vars_ = X_feat.var(numeric_only=True)
        keep = vars_[vars_ > var_thresh].index.tolist()
        if keep:
            X_feat = X_feat[keep]

    if y is not None:
        X_feat["target"] = y.values

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    X_feat.to_parquet(output_path, index=False)
    print(f"[features] wrote {output_path} shape={X_feat.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["build"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    P = load_params()
    build_features(args.input, args.output, P)


if __name__ == "__main__":
    main()
