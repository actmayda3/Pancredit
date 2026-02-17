# 🏦 Pancredit – Credit Scoring de Originación

Pancredit fianaciaminto inteligente para decisiones inteligentes: ofrecemos un sistema de score de crédito para originación, con análisis predictivo y explicaciones claras que ayudan al banco aprobar o rechazar solicitudes con mayor seguridad y agilidad y a los clientes a tomar decisiones inteligentes.
Es una aplicación de scoring crediticio desarrollada en Python y desplegada en Streamlit Cloud, diseñada para apoyar procesos de originación mediante análisis predictivo y reglas de negocio automatizadas.

---

## 🚀 Demo en vivo

👉 https://pancredit-egk2jytosnxt9y5qokez5k.streamlit.app

---

## 🧠 Modelo de Machine Learning

El sistema utiliza un modelo supervisado calibrado para estimar la probabilidad de incumplimiento (default).

- Logistic Regression calibrada
- Optimización de umbral por F1 y KS
- Pipeline con ColumnTransformer
- Manejo de desbalance de clases
- Evaluación con:
  - AUC
  - F1 Score
  - KS
  - Brier Score

---

## 📊 Funcionalidades

- Búsqueda de cliente por NIT
- Cálculo de probabilidad de default
- Asignación automática de bucket de riesgo
- Aplicación de reglas de negocio
- Perfilamiento crediticio
- Recomendaciones personalizadas

---

## 🛠 Tecnologías

- Python 3.10
- Scikit-learn 1.2.2
- Imbalanced-learn 0.10.1
- Streamlit
- GitHub
- OpenAI API

---

## 🔐 Seguridad

Las claves API están protegidas mediante Streamlit Secrets y no se almacenan en el repositorio.

---

## 📌 Autor

**Mayra López Mejía**  
Proyecto Final – Diplomado en Ciencia de Datos
