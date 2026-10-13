"""Generate synthetic transaction data V2 — 2100 rows, 29 features."""
import numpy as np
import pandas as pd

np.random.seed(42)
N_NORMAL, N_ANOMALY = 2000, 100

# Normal
amount = np.clip(np.random.lognormal(12.2, 0.8, N_NORMAL) *
    np.where(np.random.randint(1,31,N_NORMAL) > 25, 1.3, 1.0), 1000, 15_000_000)
hour = np.clip(np.concatenate([
    np.random.normal(12, 3, int(N_NORMAL*0.6)),
    np.random.normal(20, 2, N_NORMAL - int(N_NORMAL*0.6))]), 0, 23)
np.random.shuffle(hour)

normal = pd.DataFrame({
    'amount': amount,
    'duration_sec': np.clip(np.random.gamma(3, 15, N_NORMAL), 1, 300),
    'hour': hour,
    'num_items': np.clip(np.random.poisson(2.5, N_NORMAL), 1, 20),
    'day_of_week': np.random.randint(0, 7, N_NORMAL),
    'account_age_days': np.clip(np.random.exponential(365, N_NORMAL), 1, 3650),
    'num_prev_tx': np.clip(np.random.poisson(15, N_NORMAL), 0, 200),
    'payment_method': np.random.choice([0,1,2,3], N_NORMAL, p=[0.2,0.35,0.30,0.15]),
    'merchant_category': np.random.choice(range(10), N_NORMAL),
    'is_anomaly': 0
})
normal['is_weekend'] = (normal['day_of_week'] >= 5).astype(int)

# Anomaly (5 tipe)
n_a, n_b, n_c, n_d = 30, 20, 20, 15
n_e = N_ANOMALY - n_a - n_b - n_c - n_d

anomaly = pd.concat([
    pd.DataFrame({
        'amount': np.clip(np.random.lognormal(15.5, 0.5, n_a), 1000, 50_000_000),
        'duration_sec': np.random.gamma(2, 10, n_a),
        'hour': np.random.choice([1,2,3,4,5], n_a),
        'num_items': np.random.randint(1, 4, n_a),
        'day_of_week': np.random.randint(0, 7, n_a),
        'account_age_days': np.clip(np.random.exponential(30, n_a), 1, 90),
        'num_prev_tx': np.random.poisson(2, n_a),
        'payment_method': np.random.choice([1,3], n_a),
        'merchant_category': np.random.choice([7,8,9], n_a),
        'is_anomaly': 1
    }),
    pd.DataFrame({
        'amount': np.random.uniform(1000, 15000, n_b),
        'duration_sec': np.random.uniform(0.5, 3, n_b),
        'hour': np.random.choice([0,1,2,3,4,23], n_b),
        'num_items': np.ones(n_b, dtype=int),
        'day_of_week': np.random.randint(0, 7, n_b),
        'account_age_days': np.clip(np.random.exponential(15, n_b), 1, 60),
        'num_prev_tx': np.random.poisson(30, n_b),
        'payment_method': np.random.choice([1,2], n_b),
        'merchant_category': np.random.choice([0,1,2], n_b),
        'is_anomaly': 1
    }),
    pd.DataFrame({
        'amount': np.clip(np.random.normal(500_000, 50_000, n_c), 1000, None),
        'duration_sec': np.random.uniform(0.1, 1.0, n_c),
        'hour': np.full(n_c, 3),
        'num_items': np.full(n_c, 3),
        'day_of_week': np.full(n_c, 2),
        'account_age_days': np.clip(np.random.exponential(10, n_c), 1, 30),
        'num_prev_tx': np.random.poisson(50, n_c),
        'payment_method': np.full(n_c, 3),
        'merchant_category': np.full(n_c, 5),
        'is_anomaly': 1
    }),
    pd.DataFrame({
        'amount': np.random.lognormal(14, 0.8, n_d),
        'duration_sec': np.random.gamma(1, 60, n_d),
        'hour': np.random.choice([1,2,3,4,5,22,23], n_d),
        'num_items': np.random.randint(10, 50, n_d),
        'day_of_week': np.random.randint(0, 7, n_d),
        'account_age_days': np.random.uniform(365, 2000, n_d),
        'num_prev_tx': np.random.poisson(20, n_d),
        'payment_method': np.random.choice([0,1,2,3], n_d),
        'merchant_category': np.random.choice([5,6,7,8,9], n_d),
        'is_anomaly': 1
    }),
    pd.DataFrame({
        'amount': np.concatenate([np.zeros(n_e//2), np.full(n_e-n_e//2, 99_999_999)]),
        'duration_sec': np.concatenate([np.zeros(n_e//2), np.full(n_e-n_e//2, 999)]),
        'hour': np.random.uniform(0, 23, n_e),
        'num_items': np.zeros(n_e, dtype=int),
        'day_of_week': np.random.randint(0, 7, n_e),
        'account_age_days': np.random.uniform(1, 3650, n_e),
        'num_prev_tx': np.random.poisson(10, n_e),
        'payment_method': np.random.choice([0,1,2,3], n_e),
        'merchant_category': np.random.randint(0, 10, n_e),
        'is_anomaly': 1
    })
], ignore_index=True)
anomaly['is_weekend'] = np.random.randint(0, 2, len(anomaly))

df = pd.concat([normal, anomaly]).sample(frac=1, random_state=42).reset_index(drop=True)
for col, lo, hi in [('amount',0,100_000_000), ('duration_sec',0,999),
                     ('hour',0,23), ('num_items',0,500),
                     ('account_age_days',1,3650), ('num_prev_tx',0,500)]:
    df[col] = df[col].clip(lo, hi)

df.to_csv("data/transactions.csv", index=False)
print(f"✅ Saved: {len(df)} rows")
