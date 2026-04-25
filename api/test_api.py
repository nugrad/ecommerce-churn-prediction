import requests
import json

BASE = "http://localhost:8000"

# ── Test 1: Health ────────────────────────────────────────────────────────────
response = requests.get(f"{BASE}/health")
print("HEALTH:", response.json())

# ── Test 2: Single prediction ─────────────────────────────────────────────────
bad_customer = {
    "customer_id"           : "bad_experience_001",
    "delivery_delay_days"   : 8.0,      # very late
    "delivery_speed_days"   : 20.0,     # took 20 days
    "estimated_speed_days"  : 12.0,     # promised 12 days
    "speed_vs_promise_ratio": 0.6,      # slower than promised
    "was_late"              : 1,
    "approval_delay_hours"  : 48.0,     # very slow approval
    "slow_approval"         : 1,
    "has_review"            : 1,
    "review_score_filled"   : 1.0,      # worst review
    "low_review"            : 1,
    "high_review"           : 0,
    "purchase_hour"         : 22,
    "purchase_dayofweek"    : 6,
    "purchase_month"        : 12,
    "is_weekend"            : 1,
    "is_business_hours"     : 0,
    "state_churn_rate"      : 0.98,
    "is_sao_paulo"          : 0,
    "is_remote_state"       : 1
}

good_customer = {
    "customer_id"           : "good_experience_002",
    "delivery_delay_days"   : -3.0,     # arrived early
    "delivery_speed_days"   : 4.0,      # very fast
    "estimated_speed_days"  : 7.0,      # promised 7 days
    "speed_vs_promise_ratio": 1.75,     # faster than promised
    "was_late"              : 0,
    "approval_delay_hours"  : 1.5,      # instant approval
    "slow_approval"         : 0,
    "has_review"            : 1,
    "review_score_filled"   : 5.0,      # perfect review
    "low_review"            : 0,
    "high_review"           : 1,
    "purchase_hour"         : 11,
    "purchase_dayofweek"    : 1,
    "purchase_month"        : 5,
    "is_weekend"            : 0,
    "is_business_hours"     : 1,
    "state_churn_rate"      : 0.93,
    "is_sao_paulo"          : 1,
    "is_remote_state"       : 0
}

# Single prediction
r = requests.post(f"{BASE}/predict", json=bad_customer)
print("\nBAD CUSTOMER:")
print(json.dumps(r.json(), indent=2))

r = requests.post(f"{BASE}/predict", json=good_customer)
print("\nGOOD CUSTOMER:")
print(json.dumps(r.json(), indent=2))

# Batch prediction
r = requests.post(f"{BASE}/predict-batch", json=[bad_customer, good_customer])
print("\nBATCH RESULT:")
print(json.dumps(r.json(), indent=2))