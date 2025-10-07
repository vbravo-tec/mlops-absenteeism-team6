# 🤝 Contribuciones al Proyecto MLOps — Absenteeism at Work (Team 6)

Este documento define las reglas de colaboración para trabajar en este repositorio.  
El objetivo es mantener un flujo de trabajo ordenado, reproducible y seguro para todo el equipo.

---

## 📌 Flujo de trabajo con Git y DVC

### Convención de nombres de ramas
- `data/...` → tareas de limpieza, imputación, split.  
- `features/...` → ingeniería de variables.  
- `model/...` → experimentos y entrenamiento de modelos.  
- `infra/...` → configuración de DVC, MLflow, CI/CD.  
- `eda/...` → notebooks de análisis exploratorio.  

### Reglas básicas
- ❌ **Nunca trabajes directamente en `main`**.  
- ✅ Todos los cambios deben ir en una rama y luego integrarse mediante **Pull Request**.

### Versionado de datos
- ❌ No subas datasets directamente a GitHub.  
- ✅ Usa siempre DVC:

```bash
dvc add data/interim/absenteeism_clean_v1.csv
git add data/interim/absenteeism_clean_v1.csv.dvc
git commit -m "data(interim): primera versión de limpieza"
dvc push
git push origin data/cleaning-nulos

### 🧩 Pipeline reproducible

- Los scripts deben integrarse a través de `dvc.yaml`.
- Usa `params.yaml` para todos los parámetros de **limpieza**, **features** y **modelos**.
- ❌ **No hardcodees** valores en los scripts: todo ajuste configurable debe ir en `params.yaml`.


## 📑 Reglas del equipo

- **Ramas cortas y descriptivas**  
  - Una tarea = una rama.

- **Pull Requests obligatorios**  
  - Descripción clara de cambios.  
  - Enlace a issue o tarea si aplica.  

- **Protección de `main`**  
  - ❌ No commits directos.  
  - ✅ PR revisado y aprobado antes de mergear.  

- **Datos gestionados con DVC**  
  - Solo punteros `.dvc` en Git.  
  - Los datos reales se guardan en el remote (Google Drive).  

- **Ambiente reproducible**  
  - Instalar dependencias desde `requirements.txt` o `requirements.lock.txt`.  
  - Documentar nuevas dependencias antes de agregarlas.  

## ✅ Checklist para Pull Requests

Antes de abrir un PR, asegúrate de cumplir lo siguiente:

- [ ] La rama tiene un nombre claro (`data/...`, `features/...`, `model/...`, etc.).  
- [ ] Si se usaron datasets nuevos o derivados, están versionados con `dvc add` y `dvc push`.  
- [ ] Los parámetros nuevos/ajustados están en `params.yaml`.  
- [ ] El pipeline corre con `dvc repro` sin romper etapas anteriores.  
- [ ] Se actualizaron notebooks, scripts o módulos correspondientes en `mlops/`.  
- [ ] El PR incluye descripción clara de cambios y su propósito.  
