# 🔍 Tech Layoff Severity Predictor

A machine learning web application that estimates the **severity of tech company layoffs** based on company profile data — built with Random Forest, SHAP explainability, and Streamlit.

---

## 🎯 What This App Does

Given a company's profile (size, funding, industry, stage, location), the app predicts whether a potential layoff event would be **Low**, **Medium**, or **High** severity — and explains *why* the model made that prediction using SHAP values.

This is not a "will layoffs happen" predictor. It answers: **if layoffs were to occur at a company with this profile, how severe would they likely be?**

---

## 🚀 Live Demo

> Deployed on Streamlit Cloud — [link here after deployment]

---

## 📊 Dataset

- **Source:** Tech Layoffs dataset (Kaggle)
- **Size:** 2,412 layoff events from 2020–2025
- **Coverage:** Global tech companies across 45+ countries
- **Target variable:** Layoff severity — Low (<10%), Medium (10–30%), High (>30% workforce affected)

---

## 🧠 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Test Accuracy | ~68–69% |
| Test F1 Score (Macro) | 0.6875 |
| Explainability | SHAP TreeExplainer |

### Why is accuracy ~68% and not 85%+?

This is an honest limitation worth explaining — and it comes from the **data, not the model**.

**1. Class imbalance.** The dataset is heavily skewed toward "Low" severity layoffs. The model sees far fewer "High" severity examples to learn from, making it harder to distinguish between classes reliably.

**2. Limited signal in available features.** Company size, funding, and stage are useful signals — but layoff severity is also driven by macroeconomic conditions, leadership decisions, product failures, and competitive pressures. None of these are captured in the dataset.

**3. Small dataset.** 2,412 events spread across 3 classes, 45 countries, and 6 years is relatively small for a complex classification task. More data would directly improve performance.

**4. Inherent noise in the target.** Layoff severity percentages in public datasets are often self-reported or estimated — introducing label noise that no model can fully overcome.

Multiple approaches were tested to improve performance — class weighting, SMOTE oversampling via imbalanced-learn, XGBoost — and Random Forest with class balancing gave the best and most stable results. Pushing accuracy above 80% would require either a richer dataset or additional engineered features that are not publicly available.

**68–69% accuracy on a genuinely hard, imbalanced, real-world classification problem is honest and defensible.**

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Model | Scikit-learn (Random Forest) |
| Explainability | SHAP (TreeExplainer) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, SHAP plots |
| Model Persistence | Joblib |
| Deployment | Streamlit Cloud |

---

## 🔧 Features

- **Prediction Result** — Low / Medium / High severity with color-coded confidence card
- **Confidence Scores** — Probability breakdown across all three classes
- **Engineered Feature Summary** — Shows what the model actually sees after transformation
- **SHAP Waterfall Plot** — Explains how each feature shifted the prediction from baseline
- **SHAP Impact Chart** — Top 10 features ranked by contribution magnitude
- **Feature Table** — Full SHAP values with direction (pushed UP / pulled DOWN)
- **Input Summary** — Complete recap of user inputs

---

## 📁 Project Structure

```
tech-layoff-severity-predictor/
│
├── app.py                  # Main Streamlit application
├── best_model.pkl          # Trained Random Forest pipeline (scaler + model)
├── feature_columns.csv     # Ordered feature columns used during training
├── requirements.txt        # Python dependencies with pinned versions
└── README.md               # This file
```

## 👩‍💻 Built By

Built as a portfolio project demonstrating end-to-end ML — from raw data to deployed, explainable prediction app.

- Data cleaning & EDA → Python (Pandas, NumPy)
- Feature engineering & model training → Scikit-learn
- Explainability → SHAP
- Frontend & deployment → Streamlit
