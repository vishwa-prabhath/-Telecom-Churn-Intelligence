import sys
sys.path.insert(0, '/home/claude/telecom_churn')

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

from utils.preprocess import load_and_engineer, encode_for_model

print("=" * 60)
print("TELECOM CHURN MODEL TRAINING PIPELINE")
print("=" * 60)

# Load and engineer features
df = load_and_engineer('/home/claude/telecom_churn/data/telco_churn.csv')
X, y, feature_cols = encode_for_model(df)

print(f"\nDataset: {len(df)} customers | Churn rate: {y.mean():.1%}")
print(f"Features: {len(feature_cols)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE — train size: {len(X_train_sm)}")

# Train XGBoost (primary model)
print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=1, use_label_encoder=False,
    eval_metric='logloss', random_state=42, verbosity=0)
xgb.fit(X_train_sm, y_train_sm)

# Train LightGBM (comparison)
print("Training LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1)
lgbm.fit(X_train_sm, y_train_sm)

# Train Random Forest (baseline)
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)

# Evaluate all models
models = {'XGBoost': xgb, 'LightGBM': lgbm, 'RandomForest': rf}
results = {}

print("\n--- Model Evaluation ---")
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'auc': round(auc, 4),
        'avg_precision': round(ap, 4),
        'precision_churn': round(report['1']['precision'], 4),
        'recall_churn': round(report['1']['recall'], 4),
        'f1_churn': round(report['1']['f1-score'], 4),
        'accuracy': round(report['accuracy'], 4)
    }
    print(f"  {name}: AUC={auc:.4f} | F1(churn)={report['1']['f1-score']:.4f} | Recall={report['1']['recall']:.4f}")

# SHAP analysis on best model (XGBoost)
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# Top feature importances from SHAP
mean_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'shap_importance': mean_shap
}).sort_values('shap_importance', ascending=False)

print("\nTop 10 SHAP features:")
for _, row in shap_importance.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['shap_importance']:.4f}")

# Churn probability scores on full dataset
df['ChurnProbability'] = xgb.predict_proba(X)[:, 1]
df['ChurnPredicted'] = (df['ChurnProbability'] > 0.5).astype(int)
df['RiskTier'] = pd.cut(df['ChurnProbability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High'])

# Revenue at risk
df['RevenueAtRisk'] = df['MonthlyCharges'] * df['ChurnProbability']
total_monthly_revenue = df['MonthlyCharges'].sum()
revenue_at_risk = df.loc[df['RiskTier'] == 'High', 'MonthlyCharges'].sum()

print(f"\n--- Business Impact ---")
print(f"  Total monthly revenue:  ${total_monthly_revenue:,.0f}")
print(f"  Revenue at risk (High): ${revenue_at_risk:,.0f} ({revenue_at_risk/total_monthly_revenue:.1%})")
print(f"  High-risk customers:    {(df['RiskTier']=='High').sum()}")

# Save everything
joblib.dump(xgb, '/home/claude/telecom_churn/models/xgb_model.pkl')
joblib.dump(explainer, '/home/claude/telecom_churn/models/shap_explainer.pkl')
df.to_parquet('/home/claude/telecom_churn/data/scored_customers.parquet', index=False)

# Save metrics and SHAP importance
with open('/home/claude/telecom_churn/models/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

shap_importance.to_csv('/home/claude/telecom_churn/models/shap_importance.csv', index=False)

# Cohort analysis
cohort_stats = df.groupby('TenureCohort', observed=True).agg(
    customers=('customerID', 'count'),
    churn_rate=('ChurnBinary', 'mean'),
    avg_monthly=('MonthlyCharges', 'mean'),
    revenue_at_risk=('RevenueAtRisk', 'sum')
).reset_index()
cohort_stats.to_csv('/home/claude/telecom_churn/data/cohort_stats.csv', index=False)

# Contract analysis
contract_stats = df.groupby('Contract').agg(
    customers=('customerID', 'count'),
    churn_rate=('ChurnBinary', 'mean'),
    avg_monthly=('MonthlyCharges', 'mean'),
).reset_index()
contract_stats.to_csv('/home/claude/telecom_churn/data/contract_stats.csv', index=False)

print("\nAll artifacts saved. Training complete.")
print("=" * 60)
