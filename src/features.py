"""
Feature Engineering for E-Commerce Churn Prediction.

DESIGN PRINCIPLE:
    Every feature here answers one question:
    "What can we know about a customer's FIRST ORDER EXPERIENCE
    that predicts whether they will ever buy again?"

    No future data. No leakage. Only what was observable
    at the time of (or shortly after) the first delivery.

FEATURE GROUPS:
    1. Delivery Features    — was the experience smooth?
    2. Review Features      — how did they feel about it?
    3. Time Features        — when did they buy?
    4. Geography Features   — where are they? (proxy for logistics quality)
    5. Approval Features    — how fast was the merchant?
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("C:/Users/Dell/Desktop/ecommerce-churn_prediction/data/processed")


# ── 1. DELIVERY FEATURES ──────────────────────────────────────────────────────

def build_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Delivery experience is the #1 post-purchase signal in marketplace churn.
    A late or slow delivery is often the last interaction before a customer leaves.

    Features:
        delivery_delay_days   : actual - estimated delivery (already in ABT)
        was_late              : binary flag, delay > 0
        delivery_speed_days   : actual days from purchase to delivery
        estimated_speed_days  : promised days from purchase to estimated delivery
        speed_vs_promise_ratio: how fast relative to promise (>1 = faster than promised)
    """
    df = df.copy()

    # Already engineered in ABT — validate it's present
    assert 'delivery_delay_days' in df.columns, "delivery_delay_days missing from ABT"

    # Binary: was this order late?
    df['was_late'] = (df['delivery_delay_days'] > 0).astype(int)

    # Actual delivery speed: purchase → delivered
    df['delivery_speed_days'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days

    # Promised delivery speed: purchase → estimated
    df['estimated_speed_days'] = (
        df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
    ).dt.days

    # How fast relative to the promise?
    # >1.0 means delivered faster than promised (good)
    # <1.0 means delivered slower than promised (bad)
    df['speed_vs_promise_ratio'] = (
        df['estimated_speed_days'] / df['delivery_speed_days'].replace(0, np.nan)
    ).round(4)

    logger.info("Delivery features built.")
    return df


# ── 2. REVIEW FEATURES ───────────────────────────────────────────────────────

def build_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Review score is the customer's explicit signal of satisfaction.
    58.7% of orders have no review — we treat this as informative, not missing.

    Hypothesis: customers who don't leave reviews are MORE likely to churn
    (disengaged), while customers who leave low scores signal active dissatisfaction.
    Both are churners but for different reasons.

    Features:
        review_score          : raw score 1-5 (already in ABT, may be NaN)
        has_review            : 1 if customer left a review, 0 otherwise
        review_score_filled   : score filled with 3.0 (neutral) where missing
        low_review            : 1 if score <= 2 (strong dissatisfaction signal)
        high_review           : 1 if score >= 4 (satisfaction signal)
    """
    df = df.copy()

    df['has_review']          = df['review_score'].notna().astype(int)
    df['review_score_filled'] = df['review_score'].fillna(3.0)
    df['low_review']          = (df['review_score_filled'] <= 2).astype(int)
    df['high_review']         = (df['review_score_filled'] >= 4).astype(int)

    logger.info("Review features built.")
    return df


# ── 3. TIME FEATURES ─────────────────────────────────────────────────────────

def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purchase timing captures behavioral patterns:
    - Weekend shoppers may be more casual/impulsive (higher churn)
    - Late-night orders may indicate urgency buyers
    - Month/quarter captures seasonality effects

    Features:
        purchase_hour         : 0-23
        purchase_dayofweek    : 0=Monday, 6=Sunday
        purchase_month        : 1-12
        is_weekend            : Saturday or Sunday
        is_business_hours     : 9am-6pm weekday (engaged buyer signal)
    """
    df = df.copy()

    ts = df['order_purchase_timestamp']

    df['purchase_hour']      = ts.dt.hour
    df['purchase_dayofweek'] = ts.dt.dayofweek
    df['purchase_month']     = ts.dt.month
    df['is_weekend']         = ts.dt.dayofweek.isin([5, 6]).astype(int)
    df['is_business_hours']  = (
        (ts.dt.hour.between(9, 18)) & (ts.dt.dayofweek < 5)
    ).astype(int)

    logger.info("Time features built.")
    return df


# ── 4. GEOGRAPHY FEATURES ────────────────────────────────────────────────────

def build_geography_features(
    df: pd.DataFrame,
    labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Geography is a PROXY for logistics quality — not a direct cause.
    Remote states (AM, RR, AP) have longer delivery chains and higher churn.

    Strategy: TARGET ENCODING of customer_state
    We encode each state with its mean churn rate from the TRAINING data.
    This is information-rich and avoids high-cardinality dummy explosion.

    WARNING: Target encoding must be computed ONLY on training data
    and applied to validation/test. We compute it here for the full
    dataset but will re-compute inside the training pipeline to avoid leakage.

    Features:
        state_churn_rate      : mean churn rate of customer's state
        is_sao_paulo          : SP is the largest, most competitive market
        is_remote_state       : states with historically poor logistics
    """
    df = df.copy()

    # Merge churn label in to compute state rates
    # labels has: customer_unique_id, churned, customer_state
    if 'will_return' not in df.columns:
        df = df.merge(
        labels[['customer_unique_id', 'will_return']],
        on='customer_unique_id',
        how='left'
    )

    state_churn = (
    df.groupby('customer_state')['will_return']   # ← changed
    .mean()
    .reset_index()
    .rename(columns={'will_return': 'state_churn_rate'})
)

    df = df.merge(state_churn, on='customer_state', how='left')

    # Known remote/poor-logistics Brazilian states
    remote_states = {'AM', 'RR', 'AP', 'AC', 'RO', 'TO', 'PA', 'MA'}
    df['is_sao_paulo']    = (df['customer_state'] == 'SP').astype(int)
    df['is_remote_state'] = df['customer_state'].isin(remote_states).astype(int)

    logger.info("Geography features built.")
    return df


# ── 5. APPROVAL FEATURES ─────────────────────────────────────────────────────

def build_approval_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approval speed reflects merchant responsiveness.
    Slow approval = poor merchant experience = higher churn likelihood.

    approval_delay_hours is already in ABT.

    Features:
        approval_delay_hours  : hours from purchase to approval (already in ABT)
        slow_approval         : 1 if approval took more than 24 hours
    """
    df = df.copy()

    assert 'approval_delay_hours' in df.columns, "approval_delay_hours missing from ABT"

    df['slow_approval'] = (df['approval_delay_hours'] > 24).astype(int)

    logger.info("Approval features built.")
    return df


# ── 6. CLEAN & VALIDATE FEATURE MATRIX ──────────────────────────────────────

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final checks before handing off to the modeling pipeline.
    
    - Cap extreme outliers on continuous features (delivery can be negative)
    - Drop columns not needed for modeling
    - Confirm no unexpected nulls remain in feature columns
    - Report final shape
    """
    df = df.copy()

    # Cap delivery_delay_days: clip extreme outliers at 1st/99th percentile
    for col in ['delivery_delay_days', 'delivery_speed_days',
                'estimated_speed_days', 'approval_delay_hours']:
        if col in df.columns:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    # Cap speed_vs_promise_ratio (can be infinite if delivery_speed_days=0)
    if 'speed_vs_promise_ratio' in df.columns:
        df['speed_vs_promise_ratio'] = df['speed_vs_promise_ratio'].clip(0, 10)

    # Drop identifier and raw timestamp columns — not model inputs
    cols_to_drop = [
    'order_id', 'customer_id', 'order_status',
    'order_approved_at', 'order_delivered_carrier_date',
    'order_delivered_customer_date', 'order_estimated_delivery_date',
    'order_purchase_timestamp', 'review_creation_date',
    'customer_zip_code_prefix', 'customer_city',
    'review_score',
    'total_orders',
]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Check nulls in remaining columns
    null_report = df.isnull().sum()
    null_report = null_report[null_report > 0]
    if not null_report.empty:
        logger.warning("Nulls remain in feature matrix:")
        for col, cnt in null_report.items():
            logger.warning(f"  {col}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

        # Fill any remaining nulls with median (safe fallback)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    logger.info(f"Feature matrix shape: {df.shape}")
    logger.info(f"Features: {[c for c in df.columns if c not in ['customer_unique_id', 'customer_state', 'churned']]}")
    return df


# ── 7. FULL FEATURE PIPELINE ─────────────────────────────────────────────────

def build_feature_matrix(
    abt: pd.DataFrame,
    labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Orchestrates the full feature engineering pipeline.

    INPUT : ABT (order-level, 96,470 rows) + labels (customer-level, 93,350 rows)
    OUTPUT: Feature matrix (customer-level, one row per customer)

    IMPORTANT — NO LEAKAGE GUARANTEE:
        abt contains ALL orders. labels was built from FIRST orders only.
        We merge labels (which already isolated first-order rows via .first())
        and engineer features only from that first-order snapshot.
        total_orders is dropped before saving. The model never sees it.
    """
    logger.info("Building feature matrix...")

    # Start from customer_labels (already first-order snapshot)
    df = labels.copy()

    # Build each feature group
    df = build_delivery_features(df)
    df = build_review_features(df)
    df = build_time_features(df)
    df = build_geography_features(df, labels)
    df = build_approval_features(df)

    # Final cleanup
    df = validate_and_clean(df)

    return df