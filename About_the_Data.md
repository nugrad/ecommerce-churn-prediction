## RAW DATA GRANULARITY:
  orders table     → one row per ORDER
  customers table  → one row per CUSTOMER ID (transactional)
  reviews table    → one row per REVIEW

## PROBLEM:
  One customer_unique_id can have multiple orders.
  Customer "abc123" might have 3 rows in orders table.
  A model needs ONE row per customer.

## SOLUTION:
  We isolate ONLY the first order per customer.
  All features are derived from that first order exclusively.

## WHY FIRST ORDER ONLY:
  In production you score a customer right after
  their first delivery — before you know if they'll return.
  Second order data doesn't exist yet at scoring time.
  Using it would be data leakage.
------------------
Exactly How The First Order Was Isolated
# Step 1: Sort all orders by purchase timestamp ascending
abt.sort_values('order_purchase_timestamp')

# Step 2: Group by customer_unique_id
# .first() picks the EARLIEST row for each customer
# = their first order's data
first_orders = (
    abt.sort_values('order_purchase_timestamp')
    .groupby('customer_unique_id')
    .first()
    .reset_index()
)

# Result: 93,350 rows — one per unique customer
# Every feature built from this point uses only
# information available from that first order
---------------------------
## Feature Table — All 19 Features
# Group 1 — Delivery Features (5 features)
Source: orders_dataset.csv — timestamp columns
delivery_delay_days: order_delivered_customer_date minus order_estimated_delivery_date converted to days. Positive = late, Negative = early
was_late: Binary flag: 1 if delivery_delay_days > 0 else 0
delivery_speed_days: order_delivered_customer_date minus order_purchase_timestamp converted to days. How long delivery actually took
estimated_speed_days: order_estimated_delivery_date minus order_purchase_timestamp converted to days. How long delivery was promised to take
speed_vs_promise_ratio : estimated_speed_days divided by delivery_speed_days. Greater than 1 means faster than promised, less than 1 means slower

# Group 2 — Review Features (4 features)
Source: order_reviews_dataset.csv
review_score_filled: Raw review score 1-5. Where score is missing (customer left no review), filled with 3.0 (neutral). 41% of orders had no review
has_review:Binary flag: 1 if customer left any review, 0 if no review. Captures disengagement signal — non-reviewers churn more
high_review: Binary flag: 1 if review_score >= 4. Satisfaction confirmation signal
low_review: Binary flag: 1 if review_score <= 2. Strong dissatisfaction signal

# Group 3 — Time Features (5 features)
Source: orders_dataset.csv — order_purchase_timestamp
purchase_hour: Hour extracted from order_purchase_timestamp. Range 0-23
purchase_dayofweek: Day of week extracted. 0=Monday, 6=Sunday
purchase_month: Month extracted. Range 1-12. Captures seasonality — December gift buyers vs regular shoppers
is_weekend: Binary flag: 1 if purchase_dayofweek >= 5 (Saturday or Sunday)
is_business_hours: Binary flag: 1 if hour between 9-18 AND weekday. Deliberate buyer signal

# Group 4 — Geography Features (3 features)
Source: customers_dataset.csv
state_churn_rate: Target encoding — each state replaced with its mean will_return rate across all customers in that state. Computed from training data. Proxy for logistics quality per region,customer_state + will_return label

is_sao_paulo: Binary flag: 1 if customer_state == 'SP'. São Paulo is the largest, most competitive market with different behavior,
customer_state
is_remote_state : Binary flag: 1 if state is in set {AM, RR, AP, AC, RO, TO, PA, MA}. These states have the poorest logistics infrastructure in Brazil

# Group 5 — Approval Features (2 features)
Source: orders_dataset.csv
approval_delay_hours : order_approved_at minus order_purchase_timestamp converted to hours. Measures merchant responsiveness
slow_approval : Binary flag: 1 if approval_delay_hours > 24. Merchant took more than a full day to approveDerived from approval_

## Visual Summary — From 3 Raw Files To 1 Feature Row
orders_dataset.csv          customers_dataset.csv       order_reviews_dataset.csv
─────────────────           ─────────────────────       ─────────────────────────
order_purchase_timestamp  → purchase_hour               review_score
order_approved_at           purchase_dayofweek        → review_score_filled
order_delivered_customer  → purchase_month               has_review
order_estimated_delivery    is_weekend                   high_review
                            is_business_hours            low_review
      ↓                           ↓                           ↓
delivery_delay_days         customer_state
was_late                  → state_churn_rate
delivery_speed_days          is_sao_paulo
estimated_speed_days         is_remote_state
speed_vs_promise_ratio
approval_delay_hours
slow_approval

          ALL JOINED ON FIRST ORDER PER customer_unique_id
                              ↓
              93,350 rows × 19 features + 1 target
              (one row per unique customer, first order only)
---------------------------------------
## The Target Variable
# will_return :
Engineered
Count total orders per customer_unique_id across ALL orders. If count > 1 → will_return = 1. If count == 1 → will_return = 0. This is the ONLY place all orders are used — just to count them for the label. Features still use first order only.
-----------------------
## Interview Answer For This Question
# "how did you handle multiple orders per customer?":
The raw data is order-level — one row per order.
I resolved this to customer-level by isolating each customer's
first order using sort + groupby + first().
All 19 features are derived exclusively from that first order
because in production, that's all you have at scoring time.
The target label — will_return — is the only place I looked
at all orders, purely to count whether a second one ever existed.
This design guarantees zero data leakage
