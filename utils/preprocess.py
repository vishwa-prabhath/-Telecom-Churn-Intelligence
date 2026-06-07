import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['ChurnBinary'] = (df['Churn'] == 'Yes').astype(int)

    # Tenure cohorts
    df['TenureCohort'] = pd.cut(df['tenure'],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=['0-6 mo', '7-12 mo', '13-24 mo', '25-48 mo', '49-72 mo'])

    # Charge per month stability
    df['ChargePerMonth'] = np.where(df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])

    # Service count
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['ServiceCount'] = sum((df[c] == 'Yes').astype(int) for c in service_cols)

    # Has protection bundle
    df['HasProtection'] = ((df['OnlineSecurity'] == 'Yes') |
                           (df['DeviceProtection'] == 'Yes')).astype(int)

    # Revenue segment
    df['RevenueSegment'] = pd.cut(df['MonthlyCharges'],
        bins=[0, 35, 65, 90, 200],
        labels=['Low (<$35)', 'Mid ($35-65)', 'High ($65-90)', 'Premium (>$90)'])

    return df


def encode_for_model(df: pd.DataFrame):
    feature_cols = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'ServiceCount', 'HasProtection', 'ChargePerMonth',
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    X = df[feature_cols].copy()
    y = df['ChurnBinary'].copy()

    cat_cols = X.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y, feature_cols
