import pandas as pd
import numpy as np

np.random.seed(42)
n = 7043

gender = np.random.choice(['Male', 'Female'], n)
senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52])
dependents = np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70])
tenure = np.random.choice(range(0, 73), n)
phone = np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10])
multiple_lines = np.where(phone == 'Yes',
    np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.10]),
    'No phone service')
internet = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])

def inet_feat():
    return np.where(internet == 'No', 'No internet service',
                    np.random.choice(['Yes', 'No'], n, p=[0.50, 0.50]))

online_security = inet_feat()
online_backup = inet_feat()
device_protection = inet_feat()
tech_support = inet_feat()
streaming_tv = inet_feat()
streaming_movies = inet_feat()

contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.24, 0.21])
paperless = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
payment = np.random.choice(
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    n, p=[0.34, 0.23, 0.22, 0.21])

monthly = np.where(internet == 'No', np.random.uniform(19, 30, n),
          np.where(internet == 'DSL', np.random.uniform(45, 75, n),
                   np.random.uniform(70, 110, n)))
total = monthly * tenure + np.random.normal(0, 50, n)
total = np.clip(total, 0, None)

churn_prob = (
    0.05
    + 0.35 * (contract == 'Month-to-month').astype(float)
    + 0.08 * (internet == 'Fiber optic').astype(float)
    + 0.12 * (payment == 'Electronic check').astype(float)
    + 0.10 * (online_security == 'No').astype(float)
    + 0.08 * (tech_support == 'No').astype(float)
    - 0.20 * (tenure > 36).astype(float)
    - 0.15 * (contract == 'Two year').astype(float)
    + 0.05 * senior
)
churn_prob = np.clip(churn_prob, 0.02, 0.95)
churn = np.random.binomial(1, churn_prob, n)

df = pd.DataFrame({
    'customerID': [f'C{str(i).zfill(5)}' for i in range(n)],
    'gender': gender, 'SeniorCitizen': senior,
    'Partner': partner, 'Dependents': dependents,
    'tenure': tenure, 'PhoneService': phone,
    'MultipleLines': multiple_lines, 'InternetService': internet,
    'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
    'DeviceProtection': device_protection, 'TechSupport': tech_support,
    'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
    'Contract': contract, 'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly.round(2),
    'TotalCharges': total.round(2),
    'Churn': np.where(churn == 1, 'Yes', 'No')
})

df.to_csv('/home/claude/telecom_churn/data/telco_churn.csv', index=False)
print(f"Dataset: {len(df)} rows | churn rate: {df['Churn'].eq('Yes').mean():.1%}")
