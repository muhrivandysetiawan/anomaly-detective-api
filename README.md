
# 🔍 Anomaly Detective API v2

Deteksi fraud transaksi keuangan real-time dengan Machine Learning.
Versi 2: 29 fitur, 4 model dibandingkan, hyperparameter tuning.

## Problem
Fraud menyebabkan kerugian miliaran rupiah. Sistem manual tidak mampu
memonitor ribuan transaksi per detik. API ini mendeteksi anomali dalam < 10ms.

## Performa Model

| Metrik | Nilai |
|--------|-------|
| F1-Score | 0.9615 |
| Precision | 0.9259 |
| Recall | 1.0000 |
| ROC-AUC | 1.0000 |

## Endpoint

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Status API |
| GET | `/health` | Info model |
| POST | `/predict` | Prediksi 1 transaksi |
| POST | `/batch` | Prediksi hingga 100 transaksi |

## Input (9 fields)

```json
{
  "amount": 2500000,
  "duration_sec": 2,
  "hour": 2,
  "num_items": 0,
  "day_of_week": 3,
  "account_age_days": 15,
  "num_prev_tx": 50,
  "payment_method": 3,
  "merchant_category": 7
}
```

## Output

```json
{
  "is_anomaly": true,
  "label": "🚨 ANOMALI",
  "confidence": 0.99,
  "anomaly_score": -0.35,
  "latency_ms": 3.2,
  "risk_factors": ["transaksi_malam", "akun_baru", "transaksi_sangat_cepat"]
}
```

## Tech Stack
Model: Isolation Forest (tuned)
Features: 29 (10 original + 19 engineered)
API: FastAPI + Uvicorn
Scaler: RobustScaler

## Cara Jalankan

``` bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Struktur

``` plain
anomaly-detective-api/
├── app.py
├── model/
│   ├── isolation_forest_v2.pkl
│   ├── scaler_v2.pkl
│   └── features_v2.pkl
├── generate_data.py
├── train_model.py
├── requirements.txt
└── README.md
```

Portfolio ML Engineering — by Muh Rivandy Setiawan
