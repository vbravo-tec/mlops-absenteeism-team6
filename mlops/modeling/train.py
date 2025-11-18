import argparse
import json
from pathlib import Path
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib


class ParamLoader:
    """Encargado de leer y exponer los parámetros desde params.yaml."""

    def __init__(self, path="params.yaml"):
        self.path = path
        self.params = self._load_params()

    def _load_params(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        return self.params.get(key, default)


class DataLoader:
    """Carga los datos desde archivos Parquet y los separa en X, y."""

    @staticmethod
    def load_xy(parquet_path):
        df = pd.read_parquet(parquet_path)
        assert "target" in df.columns, "Se requiere columna 'target' en los conjuntos"
        X = df.drop(columns=["target"])
        y = df["target"]
        return X, y


class ModelFactory:
    """Crea modelos de sklearn según la configuración indicada."""

    @staticmethod
    def create(algorithm, params):
        alg = algorithm.lower()
        if alg == "logreg":
            return LogisticRegression(**params)
        elif alg in ("rf", "random_forest"):
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Modelo no soportado: {algorithm}")


class ModelTrainer:
    """Entrena, evalúa y guarda el modelo, además de registrar métricas en MLflow."""

    def __init__(self, params, train_path, test_path, model_out, metrics_out):
        self.params = params
        self.train_path = train_path
        self.test_path = test_path
        self.model_out = Path(model_out)
        self.metrics_out = Path(metrics_out)

        self.model = None
        self.metrics = {}

    def _configure_mlflow(self, alg):
        mlflow.set_tracking_uri("file:./mlruns")
        experiment = self.params.get("mlflow", {}).get("experiment", "Absenteeism_Model_Training")
        mlflow.set_experiment(experiment)
        run_name = f"{alg}_Absenteeism"
        return mlflow.start_run(run_name=run_name)

    def train_and_evaluate(self):
        X_train, y_train = DataLoader.load_xy(self.train_path)
        X_test, y_test = DataLoader.load_xy(self.test_path)

        mcfg = self.params.get("model", {})
        alg = mcfg.get("alg", "random_forest")
        model_params = mcfg.get("params", {"n_estimators": 100, "max_depth": 5})

        with self._configure_mlflow(alg):
            # Registrar parámetros
            mlflow.log_params({"alg": alg, **model_params})

            # Entrenar
            self.model = ModelFactory.create(alg, model_params)
            self.model.fit(X_train, y_train)

            # Evaluar
            y_pred = self.model.predict(X_test)
            self.metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            }

            # AUC si aplica
            if hasattr(self.model, "predict_proba"):
                try:
                    self.metrics["roc_auc"] = float(
                        roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
                    )
                except Exception:
                    pass

            # Guardar métricas y modelo
            self._save_results()

            # Log en MLflow
            mlflow.log_metrics(self.metrics)
            mlflow.sklearn.log_model(self.model, "model")
            mlflow.log_artifact(str(self.metrics_out))
            mlflow.log_artifact(str(self.model_out))

            print(f"\n[train] ✅ Modelo guardado en {self.model_out}")
            print(f"[train] 📊 Métricas: {self.metrics}")
            print("✅ Registro en MLflow completado con éxito.\n")

    def _save_results(self):
        self.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_out, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        self.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_out)


class TrainPipelineCLI:
    """Interfaz de línea de comandos para ejecutar el entrenamiento."""

    def __init__(self):
        self.args = self._parse_args()
        self.params = ParamLoader().params

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="Pipeline de entrenamiento")
        parser.add_argument("--train", required=True, help="Ruta del dataset de entrenamiento")
        parser.add_argument("--test", required=True, help="Ruta del dataset de prueba")
        parser.add_argument(
            "--model-out", required=True, help="Ruta donde se guardará el modelo entrenado"
        )
        parser.add_argument(
            "--metrics-out", required=True, help="Ruta donde se guardarán las métricas"
        )
        return parser.parse_args()

    def run(self):
        trainer = ModelTrainer(
            params=self.params,
            train_path=self.args.train,
            test_path=self.args.test,
            model_out=self.args.model_out,
            metrics_out=self.args.metrics_out,
        )
        trainer.train_and_evaluate()


def main():
    TrainPipelineCLI().run()


if __name__ == "__main__":
    main()
