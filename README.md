# 📡 Telecom Customer Churn Analysis

**End-to-end ML pipeline with interactive Plotly Dash dashboard**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.ai)
[![Dash](https://img.shields.io/badge/Plotly_Dash-2.x-purple)](https://dash.plotly.com)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io)

---

## Problem Statement

Telecom companies lose 15–25% of their customers annually to churn.
Each churned customer costs **$200–$400** to replace vs **$20–$30** to retain.
This project builds a production-grade churn prediction system that:

1. Identifies **which customers** will churn (ML model)
2. Explains **why** they will churn (SHAP analysis)
3. Quantifies **revenue at risk** (business impact)
4. Recommends **specific retention actions** per customer segment

---

## Architecture

```
telco_churn.csv
      │
      ▼
┌─────────────────┐
│  Feature Eng.   │  Tenure cohorts, service bundles, charge ratios
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SMOTE Balance  │  Handles 28% / 72% class imbalance
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Model Training                 │
│  • XGBoost  (AUC: 0.752)       │
│  • LightGBM (AUC: 0.755)       │
│  • RandomForest (AUC: 0.764)   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  SHAP Analysis  │  Feature importance → business insight
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Plotly Dash Dashboard (6 tabs)      │
│  • EDA          • Cohort Analysis   │
│  • Model Perf   • SHAP Insights     │
│  • Retention    • Customer Lookup   │
└──────────────────────────────────────┘
```

---

## Key Findings

| Driver | Churn Impact | Business Action |
|---|---|---|
| Month-to-month contract | +35% churn risk | Offer annual plan discount in month 1–3 |
| Fiber optic, no tech support | +18% churn risk | Bundle free tech support for 3 months |
| Electronic check payment | +12% churn risk | Incentivise auto-payment switch ($5 credit) |
| Tenure < 6 months | +20% churn risk | Proactive onboarding programme |
| No online security | +10% churn risk | Free security trial for monthly users |

**Revenue at risk (high-risk segment): $111,943/month (24.3% of total)**

---

## Dashboard Features

| Tab | What it shows |
|---|---|
| EDA | Churn by contract, internet service, payment method, monthly charges distribution |
| Cohort Analysis | Churn rate and revenue risk by tenure cohort; service count vs churn |
| Model Performance | XGBoost vs LightGBM vs RandomForest across AUC, F1, Recall, Precision |
| SHAP Analysis | Feature importance + plain-English business interpretation of each driver |
| Retention Actions | Prioritised segment strategies with customer count and revenue impact |
| Customer Lookup | Per-customer risk score, profile, and recommended action |

---

## Getting Started

```bash
# Clone and install
git clone https://github.com/vishwa-prabhath/-Telecom-Churn-Intelligence
cd telecom-churn-analysis
pip install -r requirements.txt

# Generate dataset and train model
python data/generate_dataset.py
python train_model.py

# Launch dashboard
python dashboard/app.py
# → Open http://127.0.0.1:8050
```

## Deploy on Vercel

This repo now includes a root `app.py` entrypoint that exposes the Dash server to Vercel.

1. Push the latest changes to GitHub.
2. In Vercel, choose **Add New Project** and import the GitHub repository.
3. Let Vercel detect the Python app automatically from `requirements.txt` and `app.py`.
4. Keep the default build settings, then deploy.

Local preview with the Vercel CLI:

```bash
npm i -g vercel
vercel login
vercel dev
```

If Vercel asks for the Python entrypoint, use `app.py` at the repository root.

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.10+ |
| ML | XGBoost, LightGBM, scikit-learn |
| Imbalanced data | imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer) |
| Dashboard | Plotly Dash + Dash Bootstrap Components |
| Data | pandas, NumPy |

---

## Dataset

Built on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
schema (7,043 customers, 21 features, 28.6% churn rate).

To use the real dataset: download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle
and replace `data/telco_churn.csv`.

---

## Project Structure

```
telecom_churn/
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator
│   ├── telco_churn.csv         # Input data
│   ├── scored_customers.parquet# Model predictions
│   ├── cohort_stats.csv        # Cohort analysis output
│   └── contract_stats.csv      # Contract analysis output
├── models/
│   ├── xgb_model.pkl           # Trained XGBoost model
│   ├── shap_explainer.pkl      # SHAP TreeExplainer
│   ├── shap_importance.csv     # Feature importance table
│   └── metrics.json            # Model evaluation results
├── utils/
│   └── preprocess.py           # Feature engineering pipeline
├── dashboard/
│   └── app.py                  # Plotly Dash application
├── train_model.py              # End-to-end training pipeline
├── requirements.txt
└── README.md
```

---

*Built by Vishwa Gunathilake — [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)*
