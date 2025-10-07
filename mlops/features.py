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


def build_features(input_path, output_path, P):
    df = pd.read_parquet(input_path)

    # separar target si la hubiera
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
    if cat_cols:
        if encode == "onehot":
            transformers.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
        else:
            # fallback: sin encoding
            pass

    # Scaling
    if num_cols:
        if scale == "standard":
            transformers.append(("scale", StandardScaler(), num_cols))
        elif scale == "minmax":
            transformers.append(("scale", MinMaxScaler(), num_cols))

    if transformers:
        ct = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
        X_arr = ct.fit_transform(X)
        out_cols = []

        # nombres de columnas: intentar extraer del transformer
        try:
            out_cols = ct.get_feature_names_out().tolist()
        except Exception:
            out_cols = [f"f{i}" for i in range(X_arr.shape[1])]

        X_feat = pd.DataFrame(X_arr, columns=out_cols)
    else:
        X_feat = X.copy()

    # Variance threshold simple (manual)
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
