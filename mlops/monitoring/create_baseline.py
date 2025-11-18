# mlops/monitoring/create_baseline_ks.py
import pandas as pd
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# Cargar datos de entrenamiento
df_train = pd.read_parquet(args.train)

# Calcular estadísticas básicas por columna (baseline)
baseline = {}
for col in df_train.columns:
    if pd.api.types.is_numeric_dtype(df_train[col]):
        baseline[col] = {
            "mean": float(df_train[col].mean()),  # convertir a float nativo
            "std": float(df_train[col].std()),  # convertir a float nativo
            "min": float(df_train[col].min()),  # convertir a float nativo
            "max": float(df_train[col].max()),  # convertir a float nativo
        }

# Guardar baseline en JSON
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(baseline, f, indent=4)

print(f"Baseline guardado en {args.output}")
