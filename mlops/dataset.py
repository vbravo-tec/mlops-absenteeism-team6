"""Funciones:

--task clean: carga CSV modificado (raw), limpia simple (imputación, drop cols, outliers) y guarda Parquet en data/interim/...

--task split: divide train/test desde un Parquet procesado y guarda en data/processed/..."""


import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iqr_filter(df, k=1.5, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    mask = pd.Series(True, index=df.index)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        mask &= df[col].between(lower, upper) | df[col].isna()
    return df[mask]


def zscore_filter(df, z=3.0, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    from scipy.stats import zscore as zf  # si no quieren scipy, pueden implementar manual
    zscores = df[numeric_cols].apply(lambda s: np.abs(zf(s, nan_policy="omit")))
    mask = (zscores <= z) | zscores.isna()
    # mantener filas donde *todas* las columnas cumplen
    rowmask = mask.all(axis=1)
    keep = df.index[rowmask]
    return df.loc[keep]


def clean_task(input_path, output_path, P):
    df = pd.read_csv(input_path)
    # Config
    cfg = P.get("clean", {})
    drop_cols = cfg.get("drop_columns", []) or []
    impute = cfg.get("impute", "mean")
    out_cfg = cfg.get("outliers", {"method": "iqr", "z_thresh": 3.0})
    out_method = out_cfg.get("method", "iqr")
    z_thresh = float(out_cfg.get("z_thresh", 3.0))

    # Drop columns
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    # Impute simple num/cat
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    if impute in ("mean", "median"):
        if len(num_cols) > 0:
            if impute == "mean":
                df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            else:
                df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        if len(cat_cols) > 0:
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    elif impute == "most_frequent":
        df = df.fillna(df.mode().iloc[0])
    else:
        # no imputar
        pass

    # Outliers
    if out_method == "iqr":
        df = iqr_filter(df)
    elif out_method == "zscore":
        try:
            df = zscore_filter(df, z=z_thresh)
        except Exception:
            # fallback suave si scipy no está
            df = iqr_filter(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[clean] wrote {output_path} shape={df.shape}")


def split_task(input_path, train_path, test_path, P):
    df = pd.read_parquet(input_path)
    split_cfg = P.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    random_state = int(split_cfg.get("random_state", 42))
    stratify_flag = bool(split_cfg.get("stratify", True))

    # Heurística simple: si existe "target" úsala; si no, crea dummy
    target_col = "target" if "target" in df.columns else None
    if target_col is None:
        # dataset de UCI no trae target explícita; crear dummy (columna binaria) como placeholder
        df["target"] = (df.select_dtypes(include=[np.number]).sum(axis=1) > df.select_dtypes(include=[np.number]).sum(axis=1).median()).astype(int)
        target_col = "target"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if stratify_flag and y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_parquet(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_parquet(test_path, index=False)
    print(f"[split] wrote {train_path} {X_train.shape} and {test_path} {X_test.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["clean", "split"])
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--train")
    ap.add_argument("--test")
    args = ap.parse_args()

    P = load_params()
    if args.task == "clean":
        assert args.input and args.output, "--input/--output requeridos"
        clean_task(args.input, args.output, P)
    elif args.task == "split":
        assert args.input and args.train and args.test, "--input/--train/--test requeridos"
        split_task(args.input, args.train, args.test, P)


if __name__ == "__main__":
    main()
