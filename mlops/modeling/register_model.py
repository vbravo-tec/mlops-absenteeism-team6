import mlflow
import yaml
import sys
from mlflow.entities import ViewType


def register_best():
    """
    Busca en todos los 'runs' del experimento, encuentra el mejor
    según 'METRIC_TO_OPTIMIZE' y lo registra en el "Model Registry".
    """

    # --- 1. Configuración ---
    # Cargar params.yaml para saber el nombre del experimento
    try:
        with open("params.yaml", "r", encoding="utf-8") as f:
            P = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: No se encontró 'params.yaml'. Asegúrate de estar en el dir raíz.")
        sys.exit(1)

    MLFLOW_TRACKING_URI = "file:./mlruns"
    EXPERIMENT_NAME = P.get("mlflow", {}).get("experiment", "Absenteeism_Model_Training")
    REGISTERED_MODEL_NAME = "Absenteeism-BestModel"

    # ¡Importante! La métrica que decide cuál es el "mejor" modelo.
    # Usamos 'roc_auc' porque 'accuracy' puede ser engañoso.
    METRIC_TO_OPTIMIZE = "roc_auc"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Buscando el mejor 'run' en el experimento: '{EXPERIMENT_NAME}'")
    print(f"Métrica de optimización: {METRIC_TO_OPTIMIZE}")

    # --- 2. Buscar el mejor 'run' ---
    # Usamos 'ViewType.ACTIVE_ONLY' para ignorar 'runs' eliminados
    # Ordenamos por la métrica en orden descendente (el mejor primero)
    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=[f"metrics.{METRIC_TO_OPTIMIZE} DESC"],
        )
    except Exception as e:
        print(f"Error al buscar 'runs': {e}")
        print("Asegúrate de que MLflow esté configurado y el experimento exista.")
        sys.exit(1)

    if len(runs) == 0:
        print(f"No se encontraron 'runs' en el experimento '{EXPERIMENT_NAME}'.")
        print("Ejecuta el script 'train.py' (dvc repro train) al menos una vez.")
        sys.exit(0)

    # El mejor 'run' es el primero de la lista (índice 0)
    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_metric_value = best_run[f"metrics.{METRIC_TO_OPTIMIZE}"]

    print("\n🏆 Mejor 'run' encontrado:")
    print(f"   Run ID: {best_run_id}")
    print(f"   Métrica ({METRIC_TO_OPTIMIZE}): {best_metric_value:.4f}")
    print(f"   Algoritmo (param.alg): {best_run.get('params.alg')}")

    # --- 3. Registrar el modelo de ESE 'run' ---
    print(f"\nRegistrando este modelo como: '{REGISTERED_MODEL_NAME}'")

    # Construir la URI del artefacto del modelo (dentro del 'run')
    model_uri = print(f"runs:/{best_run_id}/model")

    # Registrarlo
    registered_model_info = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    print("\n✅ ¡Éxito! Modelo registrado.")
    print(f"   Nombre: {registered_model_info.name}")
    print(f"   Versión: {registered_model_info.version}")


if __name__ == "__main__":
    # Eliminamos la lógica de 'argparse' porque ya no la necesitamos
    register_best()
