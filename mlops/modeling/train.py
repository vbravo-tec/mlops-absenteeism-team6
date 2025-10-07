import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_xy(parquet_path):
    df = pd.read_parquet(parquet_path)
    assert "target" in df.columns, "Se requiere columna 'target' en los conjuntos"
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def get_model(alg, params):
    if alg == "logreg":
        return LogisticRegression(**params)
    elif alg in ("rf", "random_forest"):
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Modelo no soportado: {alg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()

    P = load_params()
    X_train, y_train = load_xy(args.train)
    X_test, y_test = load_xy(args.test)

    mcfg = P.get("model", {})
    alg = mcfg.get("alg", "logreg")
    params = mcfg.get("params", {"max_iter": 1000})

    mlf = P.get("mlflow", {})
    tracking_uri = mlf.get("tracking_uri", "local")
    experiment = mlf.get("experiment", "absenteeism-baseline")

    if tracking_uri != "local":
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run():
        mlflow.log_params({"alg": alg, **params})

        model = get_model(alg, params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))

        metrics = {"accuracy": acc, "f1_macro": f1}
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # log MLflow
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # guarda modelo para el artefacto del pipeline
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)

        print(f"[train] saved {args.model_out} | metrics={metrics}")


if __name__ == "__main__":
    main()
