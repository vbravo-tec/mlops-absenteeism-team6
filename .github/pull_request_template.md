# 🚀 Pull Request Template

## 📋 Descripción
Por favor, incluye un resumen claro de los cambios realizados en este PR.  
Menciona el propósito (ej. limpieza de datos, nuevas features, entrenamiento de modelo, configuración de infra, etc.).

---

## ✅ Checklist

Antes de enviar este PR, asegúrate de haber revisado lo siguiente:

- [ ] La rama tiene un nombre claro (`data/...`, `features/...`, `model/...`, `infra/...`, `eda/...`).  
- [ ] Si se usaron datasets nuevos o derivados, están versionados con `dvc add` y `dvc push`.  
- [ ] Los parámetros nuevos o modificados están definidos en `params.yaml`.  
- [ ] El pipeline corre correctamente con `dvc repro` y no rompe etapas anteriores.  
- [ ] Se actualizaron notebooks, scripts o módulos correspondientes en `mlops/`.  
- [ ] El PR incluye una descripción clara de cambios y su propósito.  

---

## 📎 Referencias
- Issues relacionados: Closes #<ID-ISSUE> (si aplica)  
- Documentación/Notas adicionales: <enlaces o referencias>  

---

## 🧑‍🤝‍🧑 Roles involucrados
- [ ] Data Engineer  
- [ ] Data Scientist  
- [ ] ML Engineer  
- [ ] Software Engineer  
- [ ] SRE  
