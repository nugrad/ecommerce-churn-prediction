"""
Batch Scoring Pipeline.

What this does:
    1. Loads the trained model
    2. Loads ALL customers from the feature matrix
    3. Scores every single one
    4. Saves a ranked CSV: who is most likely to return

When to run this:
    - Every week before marketing sends campaigns
    - As a scheduled job (cron, Airflow, GitHub Actions)
    - Output goes to CRM or email tool

Run it:
    python src/batch_score.py
"""

import pandas   as pd
import numpy    as np
import joblib
import json
import logging
from pathlib    import Path
from datetime   import datetime

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# CORRECT
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "models" / "xgb_tuned.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
DATA_PATH    = BASE_DIR / "data" / "processed" / "feature_matrix.csv"
OUTPUT_DIR   = BASE_DIR / "data" / "processed"

FEATURE_COLS = [
    'delivery_delay_days', 'approval_delay_hours',
    'was_late', 'delivery_speed_days', 'estimated_speed_days',
    'speed_vs_promise_ratio', 'has_review', 'review_score_filled',
    'low_review', 'high_review', 'purchase_hour', 'purchase_dayofweek',
    'purchase_month', 'is_weekend', 'is_business_hours',
    'state_churn_rate', 'is_sao_paulo', 'is_remote_state', 'slow_approval'
]

# ── Step 1: Load model and threshold ─────────────────────────────────────────

def load_artifacts():
    logger.info("Loading model...")
    model = joblib.load(MODEL_PATH)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    threshold = metrics['best_threshold']
    logger.info(f"Threshold: {threshold:.4f}")
    return model, threshold


# ── Step 2: Load customer data ────────────────────────────────────────────────

def load_customers():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df):,} customers")
    return df


# ── Step 3: Score every customer ──────────────────────────────────────────────

def score_customers(df: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    """
    Takes raw feature matrix.
    Adds three new columns:
        return_probability    → 0.0 to 1.0
        will_return_predicted → True/False
        risk_tier             → HIGH / MEDIUM / LOW
    """
    X     = df[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]

    df = df.copy()
    df['return_probability']    = np.round(probs, 4)
    df['will_return_predicted'] = (probs >= threshold)
    df['risk_tier']             = pd.cut(
        probs,
        bins   = [0,    0.10,   0.20,   1.01],
        labels = ['LOW', 'MEDIUM', 'HIGH']
    )

    return df


# ── Step 4: Build output report ───────────────────────────────────────────────

def build_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the columns a CRM or marketing team actually needs.
    Sorted by return_probability descending — highest value customers first.
    """
    report_cols = [
        'customer_unique_id',
        'customer_state',
        'return_probability',
        'will_return_predicted',
        'risk_tier',
        # Key features for marketing context
        'delivery_delay_days',
        'review_score_filled',
        'purchase_month',
        'state_churn_rate',
    ]

    report = (
        df[report_cols]
        .sort_values('return_probability', ascending=False)
        .reset_index(drop=True)
    )
    return report


# ── Step 5: Print summary ─────────────────────────────────────────────────────

def print_summary(report: pd.DataFrame) -> None:
    total  = len(report)
    tiers  = report['risk_tier'].value_counts()

    print("\n" + "="*50)
    print("  BATCH SCORING SUMMARY")
    print("="*50)
    print(f"  Total customers scored : {total:,}")
    print(f"\n  Risk Tier Breakdown:")
    for tier in ['HIGH', 'MEDIUM', 'LOW']:
        count = tiers.get(tier, 0)
        pct   = count / total * 100
        print(f"    {tier:<8}: {count:>6,}  ({pct:.1f}%)")
    print(f"\n  Avg return probability : {report['return_probability'].mean():.4f}")
    print(f"  Max return probability : {report['return_probability'].max():.4f}")
    print("="*50 + "\n")


# ── Step 6: Save output ───────────────────────────────────────────────────────

def save_report(report: pd.DataFrame) -> None:
    today    = datetime.today().strftime('%Y-%m-%d')
    filename = OUTPUT_DIR / f"batch_scores_{today}.csv"
    report.to_csv(filename, index=False)
    logger.info(f"Saved → {filename}")
    logger.info(f"Shape : {report.shape}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_batch_scoring():
    model, threshold = load_artifacts()
    df               = load_customers()
    scored_df        = score_customers(df, model, threshold)
    report           = build_report(scored_df)

    print_summary(report)
    save_report(report)

    # Show top 10 customers most likely to return
    print("Top 10 customers most likely to return:")
    print(report.head(10)[
        ['customer_unique_id', 'return_probability', 'risk_tier', 'customer_state']
    ].to_string(index=False))

    return report


if __name__ == "__main__":
    run_batch_scoring()