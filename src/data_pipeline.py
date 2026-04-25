"""
Data pipeline for E-Commerce Churn Prediction.
Handles loading, cleaning, merging, and churn label engineering.

CHURN DEFINITION (v2 - updated):
    Snapshot approach abandoned — 99.3% churn rate made it unlearnable.
    New definition: REPEAT PURCHASE PREDICTION
    churned = 1 → customer placed exactly ONE order (never returned)
    churned = 0 → customer placed MORE THAN ONE order
    
    Business framing: "Which customers, based on their first order 
    experience, will never buy again?" This is actionable — 
    trigger retention campaign immediately after first purchase.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("C:/Users/Dell/Desktop/ecommerce-churn_prediction/data/raw")
PROCESSED_DIR = Path("C:/Users/Dell/Desktop/ecommerce-churn_prediction/data/processed")

DATETIME_COLS = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
]


# ── 1. LOAD ────────────────────────────────────────────────────────────────────

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three raw CSVs."""
    logger.info("Loading raw data...")

    orders = pd.read_csv(
        RAW_DIR / "olist_orders_dataset.csv",
        parse_dates=DATETIME_COLS
    )
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    reviews   = pd.read_csv(
        RAW_DIR / "olist_order_reviews_dataset.csv",
        parse_dates=['review_creation_date', 'review_answer_timestamp']
    )

    logger.info(f"Orders: {orders.shape} | Customers: {customers.shape} | Reviews: {reviews.shape}")
    return orders, customers, reviews


# ── 2. AUDIT ───────────────────────────────────────────────────────────────────

def audit_data(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    reviews: pd.DataFrame
) -> None:
    """Print a structured data quality report."""
    print("=" * 60)
    print("DATA QUALITY AUDIT")
    print("=" * 60)

    for name, df in [("ORDERS", orders), ("CUSTOMERS", customers), ("REVIEWS", reviews)]:
        print(f"\n── {name} ({df.shape[0]:,} rows × {df.shape[1]} cols) ──")
        null_cols = df.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        if null_cols.empty:
            print("  Nulls: None")
        else:
            for col, cnt in null_cols.items():
                pct = cnt / len(df) * 100
                print(f"  {col}: {cnt:,} nulls ({pct:.1f}%)")
        print(f"  Duplicates: {df.duplicated().sum():,}")

    print(f"\n── DATE RANGE ──")
    print(f"  First order: {orders['order_purchase_timestamp'].min()}")
    print(f"  Last order:  {orders['order_purchase_timestamp'].max()}")

    print(f"\n── ORDER STATUS DISTRIBUTION ──")
    status = orders['order_status'].value_counts()
    for s, c in status.items():
        print(f"  {s:<15} {c:>6,}  ({c/len(orders)*100:.1f}%)")

    print(f"\n── PURCHASE FREQUENCY DISTRIBUTION ──")
    merged_temp = orders.merge(customers, on='customer_id')
    freq = merged_temp.groupby('customer_unique_id')['order_id'].count()
    print(freq.value_counts().head(8).to_string())
    print(f"\n  Customers with >1 order: {(freq > 1).sum():,} ({(freq > 1).mean():.1%})")
    print("=" * 60)


# ── 3. CLEAN ───────────────────────────────────────────────────────────────────

def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only 'delivered' orders with complete delivery timestamps.

    Why: canceled/unavailable/processing orders don't represent
    real completed purchase behavior. Delivery features (our
    strongest signals) are only valid for delivered orders.
    """
    initial = len(orders)
    df = orders[orders['order_status'] == 'delivered'].copy()
    df = df.dropna(subset=['order_delivered_customer_date'])

    dropped = initial - len(df)
    logger.info(
        f"Cleaned orders: {initial:,} → {len(df):,} "
        f"(dropped {dropped:,} non-delivered / null-delivery rows)"
    )
    return df


# ── 4. BUILD ANALYTICAL BASE TABLE ────────────────────────────────────────────

def build_abt(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    reviews: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all datasets into one Analytical Base Table (ABT).
    Granularity: one row per ORDER with customer and review info.

    Design decisions:
    - Left join reviews: not all orders have reviews — keep those orders
    - Aggregate reviews per order: take the latest if multiple exist
    - customer_unique_id is the real customer identifier across orders
    """
    logger.info("Building Analytical Base Table (ABT)...")

    # Step 1: orders + customers
    abt = orders.merge(customers, on='customer_id', how='left')

    # Step 2: aggregate reviews to order level
    review_agg = (
        reviews
        .sort_values('review_creation_date')
        .groupby('order_id')
        .agg(
            review_score         = ('review_score', 'last'),
            review_creation_date = ('review_creation_date', 'last'),
        )
        .reset_index()
    )

    # Step 3: merge reviews onto ABT
    abt = abt.merge(review_agg, on='order_id', how='left')

    # Step 4: delivery delay — positive=late, negative=early
    abt['delivery_delay_days'] = (
        abt['order_delivered_customer_date'] - abt['order_estimated_delivery_date']
    ).dt.days

    # Step 5: approval delay
    abt['approval_delay_hours'] = (
        (abt['order_approved_at'] - abt['order_purchase_timestamp'])
        .dt.total_seconds() / 3600
    ).round(2)

    logger.info(f"ABT shape: {abt.shape}")
    logger.info(f"Columns: {list(abt.columns)}")
    return abt


# ── 5. ENGINEER CHURN LABEL (v2) ──────────────────────────────────────────────

def engineer_churn_label(abt: pd.DataFrame) -> pd.DataFrame:
    """
    UPDATED: Repeat Purchase Prediction approach.

    WHY WE CHANGED THIS:
        Snapshot approach produced 99.3% churn — structurally unlearnable.
        Root cause: Olist is a marketplace, not a subscription business.
        93.6% of customers buy exactly once regardless of time window.
        No ML model can learn signal from a near-constant target.

    NEW DEFINITION:
        churned = 1 → customer placed exactly 1 order (never returned)
        churned = 0 → customer placed 2+ orders (came back)

    WHAT THE MODEL WILL LEARN:
        "Given everything we know about a customer's FIRST order
        experience — delivery speed, review score, geography — 
        how likely are they to ever buy again?"

    BUSINESS VALUE:
        Trigger retention campaigns (discount, follow-up email)
        immediately after first delivery for high-risk customers.
        This is actionable at exactly the right moment.

    GRANULARITY:
        One row per CUSTOMER (not per order).
        Features will be derived from first-order behavior only,
        so there's no data leakage from future orders.

    Returns:
        customer_level DataFrame with churn label + base features
    """
    logger.info("Engineering churn labels (v2 — repeat purchase approach)...")

    # ── Step 1: Count total orders per unique customer ──
    order_counts = (
        abt.groupby('customer_unique_id')['order_id']
        .count()
        .reset_index()
        .rename(columns={'order_id': 'total_orders'})
    )

    # ── Step 2: Isolate FIRST order per customer ──
    # Sort ascending so first() gives us the earliest order
    first_orders = (
        abt.sort_values('order_purchase_timestamp')
        .groupby('customer_unique_id')
        .first()          # first order's row for each customer
        .reset_index()
    )

    # ── Step 3: Merge order count onto first order data ──
    customer_df = first_orders.merge(order_counts, on='customer_unique_id', how='left')

    # ── Step 4: Assign churn label ──
    # NEW — will_return=1 is minority (correct for XGBoost scale_pos_weight)
    # Column renamed to 'will_return' for clarity
    customer_df['will_return'] = (customer_df['total_orders'] > 1).astype(int)

    # ── Step 5: Report ──
    churn_rate     = customer_df['will_return'].mean()
    churned_count  = customer_df['will_return'].sum()
    retained_count = (customer_df['will_return'] == 0).sum()

    logger.info(f"Total unique customers  : {len(customer_df):,}")
    logger.info(f"Will return  (1, minority) : {churned_count:,}  ({churn_rate:.1%})")
    logger.info(f"One-time     (0, majority) : {retained_count:,}  ({1 - churn_rate:.1%})")
    logger.info(f"Class ratio  : {retained_count / churned_count:.1f}:1  (majority:minority)")

    return customer_df


# ── 6. SAVE ────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, filename: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    logger.info(f"Saved → {out}")


# ── 7. FULL PIPELINE RUNNER ────────────────────────────────────────────────────

def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates the full pipeline.

    CHANGED: removed snapshot_date and churn_window_days params.
    Churn is now purely structural (1 order vs 2+ orders).
    No time-window configuration needed.

    Returns:
        (abt, customer_labels)
    """
    orders, customers, reviews = load_raw_data()
    audit_data(orders, customers, reviews)

    orders_clean    = clean_orders(orders)
    abt             = build_abt(orders_clean, customers, reviews)
    customer_labels = engineer_churn_label(abt)

    save_processed(abt,             'abt.csv')
    save_processed(customer_labels, 'customer_labels.csv')

    logger.info("Pipeline complete.")
    return abt, customer_labels


if __name__ == "__main__":
    abt, labels = run_pipeline()












#    """
# Data pipeline for E-Commerce Churn Prediction.
# Production-grade: deterministic paths, validation, logging, safe aggregation.
# """

# import pandas as pd
# import logging
# from pathlib import Path

# # ── LOGGING ────────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s"
# )
# logger = logging.getLogger(__name__)

# # ── PATHS (DYNAMIC, NOT HARDCODED) ─────────────────────────────────────────
# BASE_DIR = Path(__file__).resolve().parents[1]

# RAW_DIR = BASE_DIR / "data" / "raw"
# PROCESSED_DIR = BASE_DIR / "data" / "processed"

# FILES = {
#     "orders": "olist_orders_dataset.csv",
#     "customers": "olist_customers_dataset.csv",
#     "reviews": "olist_order_reviews_dataset.csv"
# }

# DATETIME_COLS = [
#     "order_purchase_timestamp",
#     "order_approved_at",
#     "order_delivered_carrier_date",
#     "order_delivered_customer_date",
#     "order_estimated_delivery_date",
# ]

# # ── VALIDATION ─────────────────────────────────────────────────────────────

# def validate_raw_files():
#     logger.info(f"Validating RAW_DIR: {RAW_DIR}")

#     if not RAW_DIR.exists():
#         raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

#     for name, fname in FILES.items():
#         fpath = RAW_DIR / fname
#         if not fpath.exists():
#             raise FileNotFoundError(f"Missing {name} file: {fpath}")
#         logger.info(f"✓ Found {name}: {fpath}")


# def validate_schema(orders, customers, reviews):
#     required_orders_cols = [
#         "order_id", "customer_id", "order_purchase_timestamp"
#     ]

#     for col in required_orders_cols:
#         if col not in orders.columns:
#             raise ValueError(f"Missing column in orders: {col}")


# # ── 1. LOAD ────────────────────────────────────────────────────────────────

# def load_raw_data():
#     logger.info("Loading raw data...")
#     logger.info(f"BASE_DIR: {BASE_DIR}")
#     logger.info(f"RAW_DIR: {RAW_DIR}")

#     validate_raw_files()

#     orders_path = RAW_DIR / FILES["orders"]
#     customers_path = RAW_DIR / FILES["customers"]
#     reviews_path = RAW_DIR / FILES["reviews"]

#     orders = pd.read_csv(
#         orders_path,
#         parse_dates=DATETIME_COLS,
#         infer_datetime_format=True
#     )

#     customers = pd.read_csv(customers_path)

#     reviews = pd.read_csv(
#         reviews_path,
#         parse_dates=["review_creation_date", "review_answer_timestamp"],
#         infer_datetime_format=True
#     )

#     validate_schema(orders, customers, reviews)

#     logger.info(f"Orders: {orders.shape}")
#     logger.info(f"Customers: {customers.shape}")
#     logger.info(f"Reviews: {reviews.shape}")

#     return orders, customers, reviews


# # ── 2. AUDIT ───────────────────────────────────────────────────────────────

# def audit_data(orders, customers, reviews):
#     print("=" * 60)
#     print("DATA QUALITY AUDIT")
#     print("=" * 60)

#     for name, df in [("ORDERS", orders), ("CUSTOMERS", customers), ("REVIEWS", reviews)]:
#         print(f"\n── {name} ({df.shape[0]:,} rows × {df.shape[1]} cols) ──")

#         nulls = df.isnull().sum()
#         nulls = nulls[nulls > 0]

#         if nulls.empty:
#             print("  Nulls: None")
#         else:
#             for col, cnt in nulls.items():
#                 print(f"  {col}: {cnt:,} ({cnt/len(df):.1%})")

#         print(f"  Duplicates: {df.duplicated().sum():,}")

#     print("\n── DATE RANGE ──")
#     print(orders["order_purchase_timestamp"].min())
#     print(orders["order_purchase_timestamp"].max())

#     print("\n── ORDER STATUS ──")
#     print(orders["order_status"].value_counts())

#     # SAFE aggregation
#     merged = orders.merge(customers, on="customer_id")
#     freq = merged.groupby("customer_unique_id")["order_id"].nunique()

#     print("\n── PURCHASE FREQUENCY ──")
#     print(freq.value_counts().head(10))
#     print(f"\nCustomers >1 order: {(freq > 1).sum()} ({(freq > 1).mean():.1%})")


# # ── 3. CLEAN ───────────────────────────────────────────────────────────────

# def clean_orders(orders):
#     initial = len(orders)

#     df = orders[orders["order_status"] == "delivered"].copy()
#     df = df.dropna(subset=["order_delivered_customer_date"])

#     logger.info(f"Cleaned orders: {initial} → {len(df)}")

#     return df


# # ── 4. BUILD ABT ───────────────────────────────────────────────────────────

# def build_abt(orders, customers, reviews):
#     logger.info("Building ABT...")

#     abt = orders.merge(customers, on="customer_id", how="left")

#     # review aggregation (latest)
#     review_agg = (
#         reviews.sort_values("review_creation_date")
#         .groupby("order_id")
#         .agg(
#             review_score=("review_score", "last"),
#             review_creation_date=("review_creation_date", "last"),
#         )
#         .reset_index()
#     )

#     abt = abt.merge(review_agg, on="order_id", how="left")

#     # feature engineering
#     abt["delivery_delay_days"] = (
#         abt["order_delivered_customer_date"]
#         - abt["order_estimated_delivery_date"]
#     ).dt.days

#     abt["approval_delay_hours"] = (
#         (abt["order_approved_at"] - abt["order_purchase_timestamp"])
#         .dt.total_seconds() / 3600
#     )

#     logger.info(f"ABT shape: {abt.shape}")
#     return abt


# # ── 5. CHURN LABEL ─────────────────────────────────────────────────────────

# def engineer_churn_label(abt, snapshot_date="2018-04-01", churn_window_days=90):
#     snapshot = pd.Timestamp(snapshot_date)
#     pred_end = snapshot + pd.Timedelta(days=churn_window_days)

#     logger.info(f"Snapshot: {snapshot} | Window: {churn_window_days} days")

#     obs = abt[abt["order_purchase_timestamp"] < snapshot]
#     pred = abt[
#         (abt["order_purchase_timestamp"] >= snapshot)
#         & (abt["order_purchase_timestamp"] < pred_end)
#     ]

#     retained_ids = set(pred["customer_unique_id"].unique())

#     customer_df = (
#         obs.groupby("customer_unique_id")
#         .agg(
#             last_order_date=("order_purchase_timestamp", "max"),
#             first_order_date=("order_purchase_timestamp", "min"),
#             total_orders=("order_id", "nunique"),  # FIXED
#         )
#         .reset_index()
#     )

#     customer_df["recency_days"] = (
#         snapshot - customer_df["last_order_date"]
#     ).dt.days

#     customer_df["churned"] = (
#         ~customer_df["customer_unique_id"].isin(retained_ids)
#     ).astype(int)

#     logger.info(f"Churn rate: {customer_df['churned'].mean():.2%}")

#     return customer_df


# # ── 6. SAVE ────────────────────────────────────────────────────────────────

# def save_processed(df, filename):
#     PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
#     path = PROCESSED_DIR / filename
#     df.to_csv(path, index=False)
#     logger.info(f"Saved: {path}")


# # ── 7. RUN PIPELINE ────────────────────────────────────────────────────────

# def run_pipeline(snapshot_date="2018-04-01", churn_window_days=90, save=True):
#     orders, customers, reviews = load_raw_data()

#     audit_data(orders, customers, reviews)

#     orders = clean_orders(orders)
#     abt = build_abt(orders, customers, reviews)

#     labels = engineer_churn_label(abt, snapshot_date, churn_window_days)

#     if save:
#         save_processed(abt, "abt.csv")
#         save_processed(labels, "customer_labels.csv")

#     logger.info("Pipeline complete.")

#     return abt, labels


# if __name__ == "__main__":
#     run_pipeline() 