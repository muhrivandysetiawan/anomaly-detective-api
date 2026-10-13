"""Train Isolation Forest V2 — with split, tuning, evaluation."""
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

df = pd.read_csv("data/transactions.csv")

# Feature engineering
df['amount_log'] = np.log1p(df['amount'])
df['amount_per_item'] = df['amount'] / (df['num_items'] + 1)
df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
df['is_low_amount'] = (df['amount'] < df['amount'].quantile(0.05)).astype(int)
df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
df['is_late_night'] = ((df['hour'] >= 0) & (df['hour'] < 5)).astype(int)
df['is_fast_tx'] = (df['duration_sec'] < 5).astype(int)
df['is_slow_tx'] = (df['duration_sec'] > 120).astype(int)
df['amount_per_second'] = df['amount'] / (df['duration_sec'] + 0.1)
df['is_new_account'] = (df['account_age_days'] < 30).astype(int)
df['is_old_account'] = (df['account_age_days'] > 1000).astype(int)
df['tx_frequency'] = df['num_prev_tx'] / (df['account_age_days'] + 1)
df['is_high_frequency'] = (df['tx_frequency'] > df['tx_frequency'].quantile(0.90)).astype(int)
df['risk_score_raw'] = (df['is_night']*2 + df['is_new_account']*2 +
                         df['is_fast_tx'] + df['is_high_amount'] + df['is_high_frequency'])
df['night_new_account'] = df['is_night'] * df['is_new_account']
df['night_high_amount'] = df['is_night'] * df['is_high_amount']
df['fast_many_items'] = df['is_fast_tx'] * (df['num_items'] > 5).astype(int)
df['items_per_second'] = df['num_items'] / (df['duration_sec'] + 0.1)

FEATURES = [c for c in df.columns if c != 'is_anomaly']
X = df[FEATURES].values
y = df['is_anomaly'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_samples': ['auto', 0.75],
    'contamination': [0.04, 0.05, 0.06],
    'max_features': [0.75, 1.0],
}
best_score, best_params = 0, {}
for params in ParameterGrid(param_grid):
    scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for tr, val in skf.split(X_train_scaled, y_train):
        m = IsolationForest(**params, random_state=42, n_jobs=-1)
        m.fit(X_train_scaled[tr])
        yp = np.where(m.predict(X_train_scaled[val]) == -1, 1, 0)
        scores.append(f1_score(y_train[val], yp))
    avg = np.mean(scores)
    if avg > best_score:
        best_score, best_params = avg, params

# Final model
model = IsolationForest(**best_params, random_state=42, n_jobs=-1)
model.fit(X_train_scaled)

y_pred = np.where(model.predict(X_test_scaled) == -1, 1, 0)
print(f"F1={f1_score(y_test, y_pred):.4f}")
print(f"Precision={precision_score(y_test, y_pred):.4f}")
print(f"Recall={recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC={roc_auc_score(y_test, model.decision_function(X_test_scaled)):.4f}")
print(f"CM:\n{confusion_matrix(y_test, y_pred)}")

os.makedirs("model", exist_ok=True)
with open("model/isolation_forest.pkl", "wb") as f: pickle.dump(model, f)
with open("model/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
with open("model/features.pkl", "wb") as f: pickle.dump(FEATURES, f)
print("✅ Model saved")
