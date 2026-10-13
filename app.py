
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import time
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load model v2 (29 fitur) + scaler + feature list
with open("model/isolation_forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/features.pkl", "rb") as f:
    FEATURES = pickle.load(f)

logger.info(f"✅ Model loaded. Expecting {len(FEATURES)} features.")

# ── Schema input: 9 field saja (sisanya di-engineer otomatis)
class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0, le=100_000_000)
    duration_sec: float = Field(..., gt=0, le=999)
    hour: float = Field(..., ge=0, le=23)
    num_items: int = Field(..., ge=0, le=500)
    day_of_week: int = Field(..., ge=0, le=6)
    account_age_days: float = Field(..., ge=1, le=3650)
    num_prev_tx: int = Field(..., ge=0, le=500)
    payment_method: int = Field(..., ge=0, le=3)
    merchant_category: int = Field(..., ge=0, le=9)

class BatchInput(BaseModel):
    transactions: List[TransactionInput] = Field(..., max_length=100)

# ── Feature engineering: 9 input → 29 fitur (sama persis saat training)
def engineer_features(d: dict) -> np.ndarray:
    d = d.copy()

    # is_weekend dari day_of_week
    d["is_weekend"] = 1 if d["day_of_week"] >= 5 else 0

    # Fitur amount
    d["amount_log"] = np.log1p(d["amount"])
    d["amount_per_item"] = d["amount"] / (d["num_items"] + 1)
    d["is_high_amount"] = 1 if d["amount"] > 500_000 else 0
    d["is_low_amount"] = 1 if d["amount"] < 5_000 else 0

    # Fitur waktu
    d["is_night"] = 1 if (d["hour"] < 6 or d["hour"] > 22) else 0
    d["is_business_hour"] = 1 if (9 <= d["hour"] <= 17) else 0
    d["is_late_night"] = 1 if (0 <= d["hour"] < 5) else 0

    # Fitur durasi
    d["is_fast_tx"] = 1 if d["duration_sec"] < 5 else 0
    d["is_slow_tx"] = 1 if d["duration_sec"] > 120 else 0
    d["amount_per_second"] = d["amount"] / (d["duration_sec"] + 0.1)

    # Fitur akun
    d["is_new_account"] = 1 if d["account_age_days"] < 30 else 0
    d["is_old_account"] = 1 if d["account_age_days"] > 1000 else 0
    d["tx_frequency"] = d["num_prev_tx"] / (d["account_age_days"] + 1)
    d["is_high_frequency"] = 1 if d["tx_frequency"] > 0.05 else 0

    # Fitur interaksi
    d["risk_score_raw"] = (
        d["is_night"] * 2 + d["is_new_account"] * 2 +
        d["is_fast_tx"] + d["is_high_amount"] + d["is_high_frequency"]
    )
    d["night_new_account"] = d["is_night"] * d["is_new_account"]
    d["night_high_amount"] = d["is_night"] * d["is_high_amount"]
    d["fast_many_items"] = d["is_fast_tx"] * (1 if d["num_items"] > 5 else 0)
    d["items_per_second"] = d["num_items"] / (d["duration_sec"] + 0.1)

    # Susun dalam urutan FEATURES yang sama saat training
    arr = np.array([[d[f] for f in FEATURES]])
    return arr  # shape: (1, 29)

def get_risk_factors(d: dict) -> List[str]:
    factors = []
    if d["hour"] < 6 or d["hour"] > 22:
        factors.append("🌙 transaksi_malam")
    if d["account_age_days"] < 30:
        factors.append("👶 akun_baru")
    if d["duration_sec"] < 5:
        factors.append("⚡ transaksi_sangat_cepat")
    if d["amount"] > 500_000:
        factors.append("💰 nominal_besar")
    if d["num_items"] == 0:
        factors.append("📦 tanpa_item")
    if d["num_items"] > 50:
        factors.append("📦 item_terlalu_banyak")
    if d["num_prev_tx"] > 30:
        factors.append("🔄 frekuensi_tinggi")
    return factors

def predict_one(data: TransactionInput) -> dict:
    start = time.time()

    # 9 input → 29 fitur → scale → predict
    features = engineer_features(data.dict())        # (1, 29)
    features_scaled = scaler.transform(features)     # (1, 29) scaled

    score = float(model.decision_function(features_scaled)[0])
    is_anom = bool(model.predict(features_scaled)[0] == -1)
    latency = round((time.time() - start) * 1000, 2)
    conf = round(min(0.99, max(0.51, 0.5 + abs(score))), 4)

    return {
        "is_anomaly": is_anom,
        "label": "🚨 ANOMALI" if is_anom else "✅ Normal",
        "confidence": conf,
        "anomaly_score": round(score, 6),
        "latency_ms": latency,
        "risk_factors": get_risk_factors(data.dict())
    }

# ── App
app = FastAPI(
    title="Anomaly Detective API v2",
    description="ML fraud detection — 29 features, Isolation Forest tuned",
    version="2.0.0"
)

@app.get("/")
def root():
    return {
        "status": "online",
        "version": "2.0.0",
        "model": "IsolationForest (tuned)",
        "features_count": len(FEATURES),
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "IsolationForest",
        "features_count": len(FEATURES),
        "features": FEATURES
    }

@app.post("/predict")
def predict(data: TransactionInput):
    try:
        return predict_one(data)
    except Exception as e:
        logger.error(f"Predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
def batch_predict(data: BatchInput):
    try:
        results = [predict_one(tx) for tx in data.transactions]
        anomalies = sum(1 for r in results if r["is_anomaly"])
        return {
            "total": len(results),
            "anomalies": anomalies,
            "normal": len(results) - anomalies,
            "anomaly_rate": round(anomalies / len(results), 4),
            "avg_latency_ms": round(
                sum(r["latency_ms"] for r in results) / len(results), 2
            ),
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
