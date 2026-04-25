"""
FastAPI Churn Prediction API — Simplified Version

HOW FASTAPI WORKS IN 4 LINES:
    1. You create an "app" object
    2. You write normal Python functions
    3. You put a decorator (@app.get or @app.post) above each function
    4. FastAPI automatically handles HTTP, JSON parsing, and validation

That's genuinely all it is.
"""

# ── IMPORTS — what each one does ──────────────────────────────────────────────

from fastapi    import FastAPI, HTTPException
# FastAPI    → the web framework. Creates your app and handles routing.
# HTTPException → how you send error responses (like 404, 400, 500)

from pydantic   import BaseModel, Field
# BaseModel  → defines the shape of incoming JSON data
# Field      → adds validation rules (min/max values, descriptions)
# When someone sends JSON to your API, Pydantic automatically validates it
# and converts it to a Python object. If the data is wrong, it auto-rejects it.

from typing     import List
# Just Python typing — List[something] means a list of that type

import numpy    as np
import pandas   as pd
import joblib
import json
import logging
from pathlib    import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── CONSTANTS ────────────────────────────────────────────────────────────────

# CORRECT — always relative to THIS file's location, works everywhere
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "models" / "xgb_tuned.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
THRESHOLD    = 0.3737

FEATURE_COLS = [
    'delivery_delay_days', 'approval_delay_hours',
    'was_late', 'delivery_speed_days', 'estimated_speed_days',
    'speed_vs_promise_ratio', 'has_review', 'review_score_filled',
    'low_review', 'high_review', 'purchase_hour', 'purchase_dayofweek',
    'purchase_month', 'is_weekend', 'is_business_hours',
    'state_churn_rate', 'is_sao_paulo', 'is_remote_state', 'slow_approval'
]


# ── LOAD MODEL ONCE AT STARTUP ────────────────────────────────────────────────
# 
# IMPORTANT CONCEPT: you load the model HERE, outside any function.
# This runs ONCE when the server starts — not on every request.
# If you loaded inside the predict function, every request would
# reload a 50MB file from disk. That would be catastrophically slow.

logger.info("Loading model...")
model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    model_metrics = json.load(f)

logger.info("Model ready.")


# ── CREATE THE APP ────────────────────────────────────────────────────────────
#
# This is your application. Everything attaches to this object.
# title and description show up in the auto-generated docs at /docs

app = FastAPI(
    title       = "Churn Prediction API",
    description = "Predicts repeat purchase probability for e-commerce customers.",
    version     = "1.0.0"
)


# ── INPUT SCHEMA ─────────────────────────────────────────────────────────────
#
# CONCEPT: Pydantic BaseModel
#
# This defines exactly what JSON the API expects to receive.
# Think of it as a contract — if the caller sends wrong data,
# FastAPI automatically returns a 422 error with a clear message.
# You never write manual validation code.
#
# Example valid JSON this model expects:
# {
#   "customer_id": "abc123",
#   "delivery_delay_days": 3.5,
#   "was_late": 1,
#   ...
# }

class CustomerInput(BaseModel):
    customer_id             : str           # who is this customer?

    # Delivery experience features
    delivery_delay_days     : float         # positive=late, negative=early
    delivery_speed_days     : float         # how long did delivery actually take
    estimated_speed_days    : float         # how long was delivery promised
    speed_vs_promise_ratio  : float         # >1 means faster than promised
    was_late                : int           # 0 or 1

    # Merchant responsiveness
    approval_delay_hours    : float         # hours from purchase to approval
    slow_approval           : int           # 0 or 1 (>24hrs = slow)

    # Review behaviour
    has_review              : int           # did they leave a review?
    review_score_filled     : float         # 1.0-5.0 (3.0 if no review)
    low_review              : int           # score <= 2
    high_review             : int           # score >= 4

    # When they bought
    purchase_hour           : int           # 0-23
    purchase_dayofweek      : int           # 0=Mon, 6=Sun
    purchase_month          : int           # 1-12
    is_weekend              : int           # 0 or 1
    is_business_hours       : int           # 0 or 1

    # Geography
    state_churn_rate        : float         # historical return rate for their state
    is_sao_paulo            : int           # 0 or 1
    is_remote_state         : int           # 0 or 1


# ── OUTPUT SCHEMA ─────────────────────────────────────────────────────────────
#
# This defines what JSON the API sends BACK.
# FastAPI uses this to validate your response too — not just the input.

class PredictionOutput(BaseModel):
    customer_id          : str
    return_probability   : float    # 0.0 to 1.0
    will_return_predicted: bool     # True if prob >= threshold
    risk_tier            : str      # HIGH / MEDIUM / LOW


# ── HELPER FUNCTION ───────────────────────────────────────────────────────────

def assign_risk_tier(prob: float) -> str:
    """
    Converts a probability into a business-friendly tier.
    
    WHY NOT JUST USE True/False?
    Binary prediction loses information. A customer at 0.19 probability
    and one at 0.05 are both predicted "won't return" — but they need
    different marketing responses. Tiers preserve that distinction.
    """
    if prob >= 0.20:
        return "LIKELY_TO_RETURN"
    elif prob >= 0.10:
        return "UNCERTAIN"
    else:
        return "AT_RISK" 


# ── ENDPOINT 1: HEALTH CHECK ──────────────────────────────────────────────────
#
# CONCEPT: @app.get("/health")
#
# The decorator tells FastAPI: "when someone sends a GET request
# to the URL /health, run the function below it."
#
# GET = just fetching data, no body sent
# POST = sending data to the server (used for predictions)
#
# Health check is standard in all production APIs.
# Monitoring tools ping /health every 30 seconds.
# If it stops responding, the system raises an alert.

@app.get("/health")
def health_check():
    """Check if API is running."""
    return {
        "status"   : "ok",
        "threshold": THRESHOLD
    }
    # FastAPI automatically converts this dict to JSON


# ── ENDPOINT 2: MODEL INFO ────────────────────────────────────────────────────

@app.get("/model-info")
def model_info():
    """Returns model evaluation metrics."""
    return model_metrics


# ── ENDPOINT 3: SINGLE PREDICTION ────────────────────────────────────────────
#
# CONCEPT: @app.post("/predict")
#
# POST because the caller is SENDING data (customer features).
# 
# The parameter "customer: CustomerInput" tells FastAPI:
# "expect a JSON body that matches CustomerInput schema"
# FastAPI automatically parses and validates the JSON.
# If validation fails, it returns a 422 error — you write zero error code.
#
# response_model=PredictionOutput tells FastAPI:
# "validate my return value against this schema before sending"

@app.post("/predict", response_model=PredictionOutput)
def predict_single(customer: CustomerInput):
    """
    Predict return probability for a single customer.
    
    Send this JSON:
    {
        "customer_id": "abc123",
        "delivery_delay_days": 3.0,
        ... all other fields
    }
    """
    try:
        # Step 1: Convert Pydantic object → dict → DataFrame
        # The model expects a DataFrame, not a Pydantic object
        feature_dict = {col: getattr(customer, col) for col in FEATURE_COLS}
        # getattr(customer, 'delivery_delay_days') is same as customer.delivery_delay_days
        # We do it in a loop to keep it clean

        X = pd.DataFrame([feature_dict])   # single row DataFrame

        # Step 2: Get probability from model
        prob = float(model.predict_proba(X)[0, 1])
        # predict_proba returns [[prob_class0, prob_class1]]
        # [0, 1] means: first row, second column (probability of class 1 = will_return)

        # Step 3: Apply threshold
        will_return = prob >= THRESHOLD

        # Step 4: Return structured response
        return PredictionOutput(
            customer_id           = customer.customer_id,
            return_probability    = round(prob, 4),
            will_return_predicted = will_return,
            risk_tier             = assign_risk_tier(prob)
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # HTTPException sends a proper error response to the caller
        # status_code=500 means "server error"
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 4: BATCH PREDICTION ─────────────────────────────────────────────
#
# Same as single predict but accepts a LIST of customers.
# This is the "real-time batch" use case — score 10-100 customers
# in one request instead of 100 separate requests.
#
# List[CustomerInput] = a JSON array of CustomerInput objects

@app.post("/predict-batch", response_model=List[PredictionOutput])
def predict_batch(customers: List[CustomerInput]):
    """
    Predict return probability for multiple customers at once.
    Maximum 500 customers per request.

    Send this JSON:
    [
        {"customer_id": "abc", "delivery_delay_days": 3.0, ...},
        {"customer_id": "def", "delivery_delay_days": -1.0, ...}
    ]
    """
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="Empty customer list.")

    if len(customers) > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Max batch size is 500. You sent {len(customers)}."
        )

    try:
        # Step 1: Build DataFrame from list of customers
        rows = [
            {col: getattr(c, col) for col in FEATURE_COLS}
            for c in customers
        ]
        X     = pd.DataFrame(rows, columns=FEATURE_COLS)

        # Step 2: Score all at once (vectorized — fast)
        probs = model.predict_proba(X)[:, 1]

        # Step 3: Build response list
        results = [
            PredictionOutput(
                customer_id           = customers[i].customer_id,
                return_probability    = round(float(probs[i]), 4),
                will_return_predicted = bool(probs[i] >= THRESHOLD),
                risk_tier             = assign_risk_tier(float(probs[i]))
            )
            for i in range(len(customers))
        ]

        return results

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))