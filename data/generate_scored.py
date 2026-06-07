import numpy as np
import pandas as pd
from pathlib import Path
from utils.preprocess import load_and_engineer

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'

src = DATA_DIR / 'telco_churn.csv'
out = DATA_DIR / 'scored_customers.parquet'

if not src.exists():
    raise SystemExit(f"Source CSV not found: {src}")

df = load_and_engineer(str(src))

# Create a demo churn probability for the dashboard (not a real model)
rng = np.random.default_rng(42)
# Base on actual churn with some noise so distributions look realistic
df['ChurnProbability'] = np.clip(df['ChurnBinary'] * 0.7 + rng.random(len(df)) * 0.3, 0.01, 0.99)

def tier(p):
    if p >= 0.6:
        return 'High'
    if p >= 0.3:
        return 'Medium'
    return 'Low'

df['RiskTier'] = df['ChurnProbability'].apply(tier)

df.to_parquet(out, index=False)
print(f'Wrote scored dataset: {out} ({len(df)} rows)')
