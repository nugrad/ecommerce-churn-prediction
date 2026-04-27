"""
Unit tests for feature engineering pipeline.
Run with: pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy  as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.features import (
    build_delivery_features,
    build_review_features,
    build_time_features,
    build_approval_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_row():
    """Minimal valid customer row for testing."""
    return pd.DataFrame([{
        'customer_unique_id'         : 'test_001',
        'order_purchase_timestamp'   : pd.Timestamp('2018-03-15 14:30:00'),
        'order_delivered_customer_date': pd.Timestamp('2018-03-22 10:00:00'),
        'order_estimated_delivery_date': pd.Timestamp('2018-03-25 00:00:00'),
        'order_approved_at'          : pd.Timestamp('2018-03-15 16:00:00'),
        'delivery_delay_days'        : -3.0,   # arrived 3 days early
        'approval_delay_hours'       : 1.5,
        'review_score'               : 5.0,
        'customer_state'             : 'SP',
        'customer_city'              : 'sao paulo',
        'will_return'                : 1,
    }])


# ── Delivery feature tests ────────────────────────────────────────────────────

def test_was_late_early_delivery(sample_row):
    """Negative delay means early delivery — was_late must be 0."""
    result = build_delivery_features(sample_row)
    assert result['was_late'].iloc[0] == 0

def test_was_late_late_delivery(sample_row):
    """Positive delay means late delivery — was_late must be 1."""
    sample_row['delivery_delay_days'] = 5.0
    result = build_delivery_features(sample_row)
    assert result['was_late'].iloc[0] == 1

def test_delivery_speed_is_positive(sample_row):
    """Delivery speed in days must always be non-negative."""
    result = build_delivery_features(sample_row)
    assert result['delivery_speed_days'].iloc[0] >= 0

def test_speed_vs_promise_ratio(sample_row):
    """Early delivery means ratio > 1 (faster than promised)."""
    result = build_delivery_features(sample_row)
    assert result['speed_vs_promise_ratio'].iloc[0] > 1.0


# ── Review feature tests ──────────────────────────────────────────────────────

def test_has_review_when_score_present(sample_row):
    result = build_review_features(sample_row)
    assert result['has_review'].iloc[0] == 1

def test_has_review_when_score_missing(sample_row):
    sample_row['review_score'] = np.nan
    result = build_review_features(sample_row)
    assert result['has_review'].iloc[0] == 0

def test_missing_review_filled_with_neutral(sample_row):
    sample_row['review_score'] = np.nan
    result = build_review_features(sample_row)
    assert result['review_score_filled'].iloc[0] == 3.0

def test_high_review_flag(sample_row):
    """Score of 5 should set high_review=1, low_review=0."""
    result = build_review_features(sample_row)
    assert result['high_review'].iloc[0] == 1
    assert result['low_review'].iloc[0]  == 0

def test_low_review_flag(sample_row):
    """Score of 1 should set low_review=1, high_review=0."""
    sample_row['review_score'] = 1.0
    result = build_review_features(sample_row)
    assert result['low_review'].iloc[0]  == 1
    assert result['high_review'].iloc[0] == 0


# ── Time feature tests ────────────────────────────────────────────────────────

def test_purchase_hour_range(sample_row):
    result = build_time_features(sample_row)
    assert 0 <= result['purchase_hour'].iloc[0] <= 23

def test_purchase_month_range(sample_row):
    result = build_time_features(sample_row)
    assert 1 <= result['purchase_month'].iloc[0] <= 12

def test_weekend_flag_thursday(sample_row):
    """2018-03-15 is a Thursday — is_weekend must be 0."""
    result = build_time_features(sample_row)
    assert result['is_weekend'].iloc[0] == 0

def test_business_hours_flag(sample_row):
    """14:30 on Thursday is business hours — flag must be 1."""
    result = build_time_features(sample_row)
    assert result['is_business_hours'].iloc[0] == 1


# ── Approval feature tests ────────────────────────────────────────────────────

def test_fast_approval_flag(sample_row):
    """1.5 hours is fast — slow_approval must be 0."""
    result = build_approval_features(sample_row)
    assert result['slow_approval'].iloc[0] == 0

def test_slow_approval_flag(sample_row):
    """48 hours is slow — slow_approval must be 1."""
    sample_row['approval_delay_hours'] = 48.0
    result = build_approval_features(sample_row)
    assert result['slow_approval'].iloc[0] == 1