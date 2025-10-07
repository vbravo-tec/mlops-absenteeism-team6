# 📂 Proyecto MLOps — Absenteeism at Work (Team 6)

Este repositorio contiene el proyecto de la asignatura de **MLOps**, utilizando el dataset *Absenteeism at Work*.  
El objetivo es aplicar prácticas de versionado de datos, exploración, limpieza y modelado de Machine Learning en equipo.

---

## 🚀 Setup del entorno

### 1. Clonar el repositorio
```bash
git clone https://github.com/vbravo-tec/mlops-absenteeism-team6.git
cd mlops-absenteeism-team6
``` 
### 2. Crear y activar entorno virtual
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
``` 

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt

``` 
### 4. Congelar dependencias (solo si instalas algo nuevo)
```bash
pip freeze > requirements.lock.txt

```
##  Estructura de datos (DVC)

El versionado de datos se realiza con DVC.
Los datasets NO se suben a GitHub, solo los punteros .dvc.

Carpetas principales:

data/
 ├── raw/        <- Datos originales y modificados (provistos por TA)
 ├── interim/    <- Datos intermedios (limpieza parcial, imputación, etc.)
 ├── processed/  <- Datos finales listos para modelado
 └── external/   <- Datos de terceros (si se integran fuentes adicionales)

Ejemplo de datasets actuales:
data/raw/Absenteeism_at_work_original.csv
data/raw/Absenteeism_at_work_modificado.csv

##  Remote de DVC (Google Drive)
El almacenamiento remoto está configurado en Google Drive.
Cada integrante debe ejecutar una sola vez:
```bash
dvc pull
```
Pasos al correr `dvc pull` por primera vez:
1. Se abrirá el navegador con la ventana de autenticación de Google.
2. Inicia sesión con tu correo institucional (el que fue autorizado).
3. Acepta los permisos.
4. DVC almacenará el token localmente (no se versiona).

##  Flujo de trabajo con datos

### 1. Obtener datasets

```bash
git pull origin main
dvc pull
```

### 2. Crear nuevas versiones (ejemplo limpieza)

```bash
# generar dataset limpio en data/interim/
python mlops/dataset.py --task clean \
    --input data/raw/work_absenteeism_modified.csv.csv \
    --output data/interim/absenteeism_clean_v1.csv

# versionar con DVC
dvc add data/interim/absenteeism_clean_v1.csv
git add data/interim/absenteeism_clean_v1.csv.dvc
git commit -m "data(interim): primera versión de limpieza"
dvc push
git push origin <mi-rama>

```
### 3. Reproducir pipeline
```bash
dvc repro

```

## Convenciones del equipo

- **Ramas**:  
  - Crear **una rama por tarea**  
    Ejemplos:  
    - `data/cleaning-nulos`  
    - `features/encoding`  
    - `model/baseline-logreg`  
  - Nunca trabajar directamente en `main`.

- **Merge a main**:  
  - Solo vía **Pull Request (PR)**.  
  - Cada PR debe incluir descripción clara de los cambios.

- **No subir datos a GitHub**:  
  - Todos los datasets deben versionarse con **DVC**.  
  - Usar siempre:  
    ```bash
    dvc add <archivo>
    dvc push
    git add <archivo>.dvc
    git commit -m "data(...): descripción"
    git push origin <mi-rama>
    ```

- **params.yaml**:  
  - Editar parámetros de limpieza, features, split y modelos aquí.  
  - **Nunca** hardcodear valores en los scripts.

- **main protegido**:  
  - No se permiten commits directos.  
  - Solo se actualiza mediante **PR revisados y aprobados**.
