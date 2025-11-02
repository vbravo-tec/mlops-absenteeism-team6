import argparse
import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
    alg = mcfg.get("alg", "random_forest")
    params = mcfg.get("params", {"n_estimators": 100, "max_depth": 5})

    # --- Configuración MLflow ---
        # --- Configuración de MLflow ---
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = P.get("mlflow", {}).get("experiment", "Absenteeism_Model_Training")
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"{alg}_Absenteeism"):
        # 🔹 1. Registrar parámetros
        mlflow.log_params({"alg": alg, **params})

        # 🔹 2. Entrenar el modelo
        model = get_model(alg, params)
        model.fit(X_train, y_train)

        # 🔹 3. Calcular métricas
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))

        # Si el modelo tiene predict_proba (ej. RandomForest)
        if hasattr(model, "predict_proba"):
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        else:
            auc = None

        metrics = {"accuracy": acc, "f1_macro": f1}
        if auc is not None:
            metrics["roc_auc"] = auc

        # 🔹 4. Guardar métricas en archivo JSON
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # 🔹 5. Guardar modelo antes de loguearlo
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)

        # 🔹 6. Registrar resultados en MLflow
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(args.metrics_out)
        mlflow.log_artifact(args.model_out)

        print(f"\n[train] ✅ saved {args.model_out}")
        print(f"[train] 📊 metrics={metrics}")
        print("✅ Registro en MLflow completado con éxito.\n")

if __name__ == "__main__":
    main()
