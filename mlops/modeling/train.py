import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
from mlops.config import PROCESSED

def load_data():
    # Reemplaza por tu archivo final procesado
    df = pd.read_csv(PROCESSED / "absenteeism_clean.csv")
    y = df["target"]
    X = df.drop(columns=["target"])
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    with mlflow.start_run():
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")
