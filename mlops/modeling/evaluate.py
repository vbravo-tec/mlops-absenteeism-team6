import argparse
import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class ModelLoader:
    """Encargado de cargar el modelo entrenado desde disco."""

    @staticmethod
    def load(model_path):
        print(f"[evaluate] Cargando modelo desde {model_path}")
        return joblib.load(model_path)


class TestDataLoader:
    """Encargado de cargar el dataset de prueba."""

    @staticmethod
    def load(test_path):
        print(f"[evaluate] Cargando datos desde {test_path}")
        df = pd.read_parquet(test_path)
        assert "target" in df.columns, "El conjunto de prueba debe incluir la columna 'target'"
        X = df.drop(columns=["target"])
        y = df["target"]
        return X, y


class ModelEvaluator:
    """Realiza la evaluación del modelo con métricas estándar."""

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.results = {}

    def evaluate(self):
        print("[evaluate] Generando predicciones...")
        y_pred = self.model.predict(self.X)

        report = classification_report(self.y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(self.y, y_pred).tolist()

        roc_auc = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(self.X)[:, 1]
                roc_auc = float(roc_auc_score(self.y, y_proba))
            except Exception:
                pass

        self.results = {
            "classification_report": report,
            "confusion_matrix": conf_matrix,
            "roc_auc": roc_auc,
        }
        return self.results

    def save_report(self, report_path):
        print(f"[evaluate] Guardando reporte en {report_path}")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"[evaluate] ✅ Reporte guardado exitosamente en {report_path}")


class EvaluationPipelineCLI:
    """Interfaz de línea de comandos para ejecutar la evaluación del modelo."""

    def __init__(self):
        self.args = self._parse_args()

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="Evaluación de modelos entrenados")
        parser.add_argument(
            "--model", required=True, help="Ruta al archivo del modelo entrenado (.joblib)"
        )
        parser.add_argument("--test", required=True, help="Ruta al dataset de prueba (.parquet)")
        parser.add_argument(
            "--report", required=True, help="Ruta donde se guardará el reporte (.json)"
        )
        return parser.parse_args()

    def run(self):
        model = ModelLoader.load(self.args.model)
        X, y = TestDataLoader.load(self.args.test)
        evaluator = ModelEvaluator(model, X, y)
        evaluator.evaluate()
        evaluator.save_report(self.args.report)


def main():
    EvaluationPipelineCLI().run()


if __name__ == "__main__":
    main()
