# --- 1. Importación de Librerías ---
# Librerías estándar para manipulación de datos, modelos y métricas
import pandas as pd
import yaml
import os
import json
import joblib # Para guardar el modelo y el escalador

# Clases de Scikit-learn para el preprocesamiento y el modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("Script de entrenamiento iniciado...")

# --- 2. Carga de Parámetros y Datos ---
# Cargar los parámetros definidos en params.yaml
try:
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    print("Parámetros cargados correctamente.")
except Exception as e:
    print(f"Error al cargar params.yaml: {e}")
    # Salir si no se pueden cargar los parámetros
    exit()

# Definir rutas de archivos
data_path = 'data/interim/absenteeism_eda_fe_intermediate.csv'
model_dir = 'models'
metrics_path = 'metrics.json'

# Crear el directorio 'models' si no existe
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'model.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib') # Ruta para guardar el escalador

# Cargar el dataset
print(f"Cargando dataset desde: {data_path}")
df = pd.read_csv(data_path)


# --- 3. Preprocesamiento y Definición de Target ---
# Replicamos los mismos pasos de preprocesamiento del notebook
print("Realizando preprocesamiento...")
median_hours = df['Absenteeism time in hours'].median()
df['Target_Binary'] = (df['Absenteeism time in hours'] > median_hours).astype(int)

y = df['Target_Binary']
X = df.drop(columns=['ID', 'Absenteeism time in hours', 'Target_Binary'])


# --- 4. División y Escalado de Datos ---
# Dividir los datos en conjuntos de entrenamiento y prueba
print("Dividiendo los datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params['train']['test_size'],
    random_state=params['base']['random_state'],
    stratify=y
)

# Escalar las características
print("Escalando características...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 5. Entrenamiento del Modelo ---
# Instanciar el modelo Random Forest con los hiperparámetros de params.yaml
print("Entrenando el modelo Random Forest...")
model = RandomForestClassifier(
    n_estimators=params['train']['n_estimators'],
    max_depth=params['train']['max_depth'],
    min_samples_leaf=params['train']['min_samples_leaf'],
    random_state=params['base']['random_state']
)

# Entrenar el modelo
model.fit(X_train_scaled, y_train)
print("Modelo entrenado exitosamente.")


# --- 6. Evaluación del Modelo ---
# Realizar predicciones y calcular métricas
print("Evaluando el modelo...")
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)


# --- 7. Guardado de Salidas (Outputs) ---
# Guardar las métricas en el archivo JSON que DVC rastrea
print(f"Guardando métricas en: {metrics_path}")
with open(metrics_path, 'w') as f:
    json.dump(
        {'accuracy': accuracy, 'f1_score': f1, 'roc_auc': roc_auc},
        f,
        indent=4
    )

# Guardar el modelo entrenado
print(f"Guardando modelo en: {model_path}")
joblib.dump(model, model_path)

# ¡Importante! Guardar también el escalador. El modelo es inútil sin el
# escalador exacto con el que fue entrenado para procesar datos nuevos.
print(f"Guardando escalador en: {scaler_path}")
joblib.dump(scaler, scaler_path)

print("\n--- Resultados Finales ---")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  ROC AUC: {roc_auc:.4f}")
print("\nScript de entrenamiento finalizado exitosamente. ¡Listo para 'dvc repro'!")