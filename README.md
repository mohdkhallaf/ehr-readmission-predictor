# 🩺 Electronic Health Records (EHR) Patient Readmission Predictor

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-red.svg)](https://shap.readthedocs.io/)

Predictive machine learning application built to estimate 30-day hospital readmission risks using clinical EHR data, leveraging SHAP for model explainability in clinical workflows.

## 📌 Features
- **Class Imbalance Handling:** XGBoost optimization for imbalanced outcome distributions.
- **Explainable AI (XAI):** SHAP waterfall plots explaining individual risk drivers per patient.
- **Interactive UI:** Streamlit interface for live risk scoring and feature impact breakdown.
```[cite: 1, 2]