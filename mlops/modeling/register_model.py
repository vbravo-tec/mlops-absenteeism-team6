import argparse
import json
import joblib
import mlflow
import mlflow.sklearn


class ModelLoader:
    """Carga el modelo desde un archivo .joblib."""

    @staticmethod
    def load(model_path):
        print(f"[register_model] Cargando modelo desde {model_path}")
        return joblib.load(model_path)


class MetricsLoader:
    """Carga las métricas generadas por el pipeline de entrenamiento o evaluación."""

    @staticmethod
    def load(metrics_path):
        print(f"[register_model] Cargando métricas desde {metrics_path}")
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)


class MLflowRegistrar:
    """Encargado de configurar MLflow y registrar el modelo en el Model Registry."""

    def __init__(self, tracking_uri="file:./mlruns", experiment_name="Absenteeism_Model_Training"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._configure_mlflow()

    def _configure_mlflow(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        print(f"[register_model] MLflow configurado con URI={self.tracking_uri}, experimento={self.experiment_name}")

    def register_model(self, model, metrics, model_name="Absenteeism-BestModel"):
        with mlflow.start_run(run_name="Register_Model"):
            # Registrar métricas
            mlflow.log_metrics(metrics)
            # Registrar el modelo
            mlflow.sklearn.log_model(model, artifact_path="model")

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri=model_uri, name=model_name)

            print(f"[register_model] ✅ Modelo registrado en MLflow Registry como '{model_name}'")


class ModelRegistrationCLI:
    """Interfaz CLI para registrar modelos entrenados en MLflow Registry."""

    def __init__(self):
        self.args = self._parse_args()

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="Registro de modelo en MLflow Registry")
        parser.add_argument("--model", required=True, help="Ruta del modelo entrenado (.joblib)")
        parser.add_argument("--metrics", required=True, help="Ruta del archivo de métricas (.json)")
        return parser.parse_args()

    def run(self):
        model = ModelLoader.load(self.args.model)
        metrics = MetricsLoader.load(self.args.metrics)
        registrar = MLflowRegistrar()
        registrar.register_model(model, metrics)


def main():
    ModelRegistrationCLI().run()


if __name__ == "__main__":
    main()
