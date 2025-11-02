import argparse
import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    # --- Cargar modelo y datos ---
    print(f"[evaluate] Cargando modelo desde {args.model}")
    model = joblib.load(args.model)
    df = pd.read_parquet(args.test)

    X = df.drop(columns=["target"])
    y = df["target"]

    # --- Evaluar modelo ---
    print("[evaluate] Generando predicciones...")
    y_pred = model.predict(X)

    report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred).tolist()

    # Calcular ROC AUC si el modelo lo soporta
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
            roc_auc = float(roc_auc_score(y, y_proba))
        except Exception:
            pass

    results = {
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "roc_auc": roc_auc,
    }

    # --- Guardar resultados ---
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[evaluate] ✅ Reporte guardado en {args.report}")


if __name__ == "__main__":
    main()
