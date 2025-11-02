import argparse
import json
import joblib
import mlflow
import mlflow.sklearn
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    print(f"[register_model] Cargando modelo desde {args.model}")
    model = joblib.load(args.model)

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # --- Configurar MLflow ---
    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Absenteeism_Model_Training")

    # --- Registrar modelo ---
    with mlflow.start_run(run_name="Register_Model"):
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_name = "Absenteeism-BestModel"
        result = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name=model_name,
        )

        print(f"[register_model] ✅ Modelo registrado en MLflow Registry como '{model_name}'")

if __name__ == "__main__":
    main()
