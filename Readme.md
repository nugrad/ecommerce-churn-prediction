# E-Commerce Churn Prediction

> End-to-end machine learning system predicting repeat purchase likelihood  
> for e-commerce customers using first-order behavioral signals.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)]()
[![Docker](https://img.shields.io/badge/Deploy-Docker-blue)]()
---

## Problem Statement

**Can we predict which customers will never buy again — before they leave?**

Given a customer's first order experience on a Brazilian e-commerce marketplace,  
predict the probability they will place a second order.

**Why this matters:**  
Acquiring a new customer costs 5-7x more than retaining one. Identifying  
high-churn-risk customers immediately after their first delivery enables  
targeted retention campaigns at exactly the right moment.
---

## Key Technical Decision — Problem Reframing

The initial snapshot-based churn definition (did the customer return  
within 90 days?) produced a **99.3% churn rate** — structurally unlearnable  
by any model. Root cause: Olist is a non-subscription marketplace where  
93.6% of customers purchase exactly once regardless of time window.

**Solution:** Reframe as **repeat purchase prediction**.  
- `will_return = 1` → customer placed 2+ orders (3% minority)  
- `will_return = 0` → customer placed exactly 1 order (97% majority)  

This is the correct framing for non-contractual churn and is directly  
actionable for post-purchase retention campaigns.

---
## Dataset

**Source:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

| File | Rows | Description |
|------|------|-------------|
| orders_dataset.csv | 99,441 | Order lifecycle and timestamps |
| customers_dataset.csv | 99,441 | Customer identity and geography |
| order_reviews_dataset.csv | 99,224 | Review scores and comments |

**After cleaning:** 93,350 unique customers · 19 engineered features

---
## Architecture
Raw CSVs (3 files)
↓
src/data_pipeline.py     — merge, clean, churn label engineering
↓
src/features.py          — 19 features across 5 groups
↓
src/train.py             — XGBoost + Optuna tuning + threshold optimization
↓
models/xgb_tuned.pkl     — serialized model artifact
↓
┌─────────────────┬──────────────────────┐
│  api/main.py    │  dashboard/app.py    │
│  FastAPI REST   │  Streamlit UI        │
│  /predict       │  3-page dashboard    │
│  /predict-batch │  single + batch      │
└─────────────────┴──────────────────────┘
↓
Dockerfile → containerized deployment
---

## Feature Engineering

| Group | Features | Signal |
|-------|----------|--------|
| Delivery | delay_days, speed_days, was_late, speed_vs_promise | Experience quality |
| Review | score, has_review, low_review, high_review | Satisfaction |
| Time | hour, dayofweek, month, is_weekend, is_business_hours | Behavioral pattern |
| Geography | state_churn_rate, is_sao_paulo, is_remote_state | Logistics proxy |
| Approval | approval_delay_hours, slow_approval | Merchant quality |

**Top features by SHAP:**

| Rank | Feature | Business Meaning |
|------|---------|-----------------|
| 1 | state_churn_rate | State logistics quality |
| 2 | purchase_month | Seasonality — holiday vs regular buyers |
| 3 | estimated_speed_days | Delivery promise length |
| 4 | approval_delay_hours | Merchant responsiveness |
| 5 | purchase_hour | Time-of-day purchase behavior |

---

## Model Performance

| Model | AUC-ROC | AUC-PR | F1 (minority) |
|-------|---------|--------|---------------|
| Baseline — Logistic Regression | 0.5345 | 0.0331 | 0.0616 |
| XGBoost (default) | 0.5845 | 0.0393 | 0.0717 |
| **XGBoost (Optuna tuned)** | **0.5743** | **0.0382** | **0.0779** |

**Why AUC-ROC of 0.58 is honest — not a failure:**

> A random classifier scores AUC-ROC = 0.50 and AUC-PR = 0.030 (base rate).  
> Our model achieves **+8.4% AUC-ROC lift** and **+30% AUC-PR lift** over random.  
> The weak signal reflects a genuine data constraint: first-order behavioral  
> features explain only a fraction of repeat purchase intent. Missing signals  
> (product category, order value, seller rating) are not available in the  
> public dataset. The model is suitable for **prioritizing** retention outreach,  
> not as a sole decision system.

---

## Running Locally

**1. Clone and set up:**
```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Run the full pipeline:**
```bash
python src/data_pipeline.py
python src/features.py
python src/train.py
```

**3. Start the API:**
```bash
uvicorn api.main:app --reload --port 8000
# Visit http://localhost:8000/docs
```

**4. Start the dashboard:**
```bash
streamlit run dashboard/app.py
# Visit http://localhost:8501
```

**5. Run batch scoring:**
```bash
python src/batch_score.py
```

**6. Run tests:**
```bash
pytest tests/ -v
```

---

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "customer_id"           : "cust_001",
        "delivery_delay_days"   : 3.0,
        "delivery_speed_days"   : 10.0,
        "estimated_speed_days"  : 7.0,
        "speed_vs_promise_ratio": 0.7,
        "was_late"              : 1,
        "approval_delay_hours"  : 2.0,
        "slow_approval"         : 0,
        "has_review"            : 1,
        "review_score_filled"   : 3.0,
        "low_review"            : 0,
        "high_review"           : 0,
        "purchase_hour"         : 14,
        "purchase_dayofweek"    : 2,
        "purchase_month"        : 6,
        "is_weekend"            : 0,
        "is_business_hours"     : 1,
        "state_churn_rate"      : 0.97,
        "is_sao_paulo"          : 0,
        "is_remote_state"       : 0
    }
)
print(response.json())
# {"customer_id": "cust_001", "return_probability": 0.12,
#  "will_return_predicted": false, "risk_tier": "UNCERTAIN"}
```

---

## Project Structure
├── src/                    # Production pipeline code
├── api/                    # FastAPI REST endpoint
├── dashboard/              # Streamlit UI
├── models/                 # Trained model artifacts
├── notebooks/              # EDA and modeling notebooks
├── tests/                  # Unit tests (pytest)
├── data/processed/         # Feature matrix and outputs
├── Dockerfile              # Container definition
└── config.yaml             # Project configuration

---

## What I Would Do With More Data

1. Add product category — strongest missing signal
2. Add order value — high-spend customers have different retention curves
3. Add seller-level features — poor seller experience drives permanent churn
4. Build a time-series model (LSTM/GRU) on customer event sequences
5. Implement proper MLflow experiment tracking
6. Add Evidently AI for data drift monitoring in production