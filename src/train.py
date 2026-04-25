"""
Model Training Pipeline for E-Commerce Churn Prediction.

TARGET VARIABLE (corrected):
    will_return = 1 → customer placed 2+ orders (minority, 3%)
    will_return = 0 → customer placed exactly 1 order (majority, 97%)

    Positive class = will_return = 1 (the MINORITY)
    This is correct for XGBoost's scale_pos_weight which amplifies
    the positive class. scale_pos_weight = majority/minority = ~32.

METRICS:
    Primary   → AUC-ROC  (honest measure, not fooled by base rate)
    Secondary → AUC-PR   (precision-recall, best for imbalanced)
    Report    → F1 on minority class (will_return=1)
    NEVER     → accuracy
"""

import pandas as pd
import numpy as np
import logging
import joblib
import json
import warnings
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report,
    precision_recall_curve, ConfusionMatrixDisplay, confusion_matrix
)
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("C:/Users/Dell/Desktop/ecommerce-churn_prediction/data/processed")
MODELS_DIR    = Path("C:/Users/Dell/Desktop/ecommerce-churn_prediction/models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET = 'will_return'

FEATURE_COLS = [
    'delivery_delay_days', 'approval_delay_hours',
    'was_late', 'delivery_speed_days', 'estimated_speed_days',
    'speed_vs_promise_ratio', 'has_review', 'review_score_filled',
    'low_review', 'high_review', 'purchase_hour', 'purchase_dayofweek',
    'purchase_month', 'is_weekend', 'is_business_hours',
    'state_churn_rate', 'is_sao_paulo', 'is_remote_state', 'slow_approval'
]


# ── 1. LOAD & SPLIT ──────────────────────────────────────────────────────────

def load_and_split(
    path: str = 'C:/Users/Dell/Desktop/ecommerce-churn_prediction/data/processed/feature_matrix.csv',
    test_size: float = 0.2,
    random_state: int = 42
):
    df = pd.read_csv(path)

    # Validate target exists
    assert TARGET in df.columns, f"'{TARGET}' column missing. Re-run pipeline."

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    minority = y_train.sum()
    majority = (y_train == 0).sum()

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train — will_return=1 (minority): {minority:,} ({y_train.mean():.3f})")
    logger.info(f"Train — will_return=0 (majority): {majority:,} ({1-y_train.mean():.3f})")
    logger.info(f"scale_pos_weight = {majority/minority:.1f}")

    return X_train, X_test, y_train, y_test


# ── 2. BASELINE — LOGISTIC REGRESSION ───────────────────────────────────────

def train_baseline(X_train, X_test, y_train, y_test) -> dict:
    logger.info("Training baseline: Logistic Regression...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            C=0.1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return _evaluate(y_test, y_pred, y_prob, "Baseline (LR)")


# ── 3. XGBOOST DEFAULT ───────────────────────────────────────────────────────

def train_xgboost(X_train, X_test, y_train, y_test) -> tuple:
    logger.info("Training XGBoost (default params)...")

    # CORRECT: minority=positive(1), majority=negative(0)
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale:.1f}  (majority/minority — correct direction)")

    model = xgb.XGBClassifier(
        n_estimators         = 500,
        learning_rate        = 0.05,
        max_depth            = 6,
        scale_pos_weight     = scale,   # now correctly amplifies minority(1)
        eval_metric          = 'aucpr',
        early_stopping_rounds= 30,
        random_state         = 42,
        n_jobs               = -1,
        verbosity            = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = _evaluate(y_test, y_pred, y_prob, "XGBoost (default)")
    return model, metrics


# ── 4. OPTUNA TUNING ─────────────────────────────────────────────────────────

def tune_xgboost(X_train, X_test, y_train, y_test, n_trials: int = 50) -> tuple:
    logger.info(f"Optuna tuning: {n_trials} trials...")

    scale = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 200, 800),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth'       : trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample'       : trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            # Let Optuna explore scale_pos_weight around the true ratio
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale * 0.5, scale * 2),
            'eval_metric'     : 'aucpr',
            'random_state'    : 42,
            'n_jobs'          : -1,
            'verbosity'       : 0,
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr  = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr  = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr, verbose=False)
            prob = m.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, prob))   # AUC-ROC in CV

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best CV AUC-ROC : {study.best_value:.4f}")
    logger.info(f"Best params     : {study.best_params}")

    best_params = study.best_params
    best_params.update({
        'eval_metric' : 'aucpr',
        'random_state': 42,
        'n_jobs'      : -1,
        'verbosity'   : 0
    })

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train, verbose=False)

    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = _evaluate(y_test, y_pred, y_prob, "XGBoost (Optuna tuned)")

    joblib.dump(best_model, MODELS_DIR / 'xgb_tuned.pkl')
    logger.info("Saved → models/xgb_tuned.pkl")

    return best_model, metrics, study


# ── 5. THRESHOLD OPTIMIZATION ────────────────────────────────────────────────

def optimize_threshold(model, X_test, y_test, beta: float = 1.0) -> float:
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    beta2    = beta ** 2
    f_scores = (
        (1 + beta2) * precision * recall /
        (beta2 * precision + recall + 1e-9)
    )

    best_idx       = np.argmax(f_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f         = f_scores[best_idx]

    logger.info(f"Optimal threshold: {best_threshold:.4f}  (F{beta}={best_f:.4f})")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(recall, precision, color='#3498db', linewidth=2, label='PR Curve')
    ax.axhline(y_test.mean(), color='gray', linestyle='--',
               linewidth=1, label=f'Baseline (random) = {y_test.mean():.3f}')
    ax.scatter(
        recall[best_idx], precision[best_idx],
        color='#e74c3c', s=150, zorder=5,
        label=f'Optimal @ {best_threshold:.3f}  F{beta}={best_f:.3f}'
    )
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('C:/Users/Dell/Desktop/ecommerce-churn_prediction/data/processed/pr_curve.png', bbox_inches='tight', dpi=150)
    plt.show()

    return best_threshold


# ── 6. CONFUSION MATRIX PLOT ─────────────────────────────────────────────────

def plot_confusion(y_test, y_pred, threshold: float) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=['One-time (0)', 'Returns (1)'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix @ threshold={threshold:.3f}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('C:/Users/Dell/Desktop/ecommerce-churn_prediction/data/processed/confusion_matrix.png', bbox_inches='tight', dpi=150)
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives  (correctly predicted one-time) : {tn:,}")
    print(f"  False Positives (predicted return, actually left): {fp:,}")
    print(f"  False Negatives (missed actual returners)        : {fn:,}")
    print(f"  True Positives  (correctly predicted return)     : {tp:,}")


# ── 7. FEATURE IMPORTANCE ────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list) -> None:
    importance = model.get_booster().get_score(importance_type='gain')
    imp_df = (
        pd.DataFrame(importance.items(), columns=['feature', 'gain'])
        .sort_values('gain', ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp_df['feature'], imp_df['gain'], color='#3498db', edgecolor='white')
    ax.set_xlabel('Gain')
    ax.set_title('XGBoost Feature Importance (Gain)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('C:/Users/Dell/Desktop/ecommerce-churn_prediction/data/processed/feature_importance.png', bbox_inches='tight', dpi=150)
    plt.show()


# ── 8. EVALUATION ────────────────────────────────────────────────────────────

def _evaluate(y_test, y_pred, y_prob, model_name: str) -> dict:
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)
    f1_min  = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  AUC-ROC         : {auc_roc:.4f}  ← primary (honest)")
    print(f"  AUC-PR          : {auc_pr:.4f}")
    print(f"  F1 (returners)  : {f1_min:.4f}  ← minority class")
    print(f"\n{classification_report(y_test, y_pred, target_names=['One-time','Returns'], zero_division=0)}")

    return {
    'model'   : model_name,
    'auc_roc' : float(auc_roc),
    'auc_pr'  : float(auc_pr),
    'f1_min'  : float(f1_min)
}


# ── 9. FULL RUNNER ───────────────────────────────────────────────────────────

def run_training(n_optuna_trials: int = 50) -> dict:
    X_train, X_test, y_train, y_test = load_and_split()

    baseline_metrics          = train_baseline(X_train, X_test, y_train, y_test)
    xgb_model, xgb_metrics   = train_xgboost(X_train, X_test, y_train, y_test)
    best_model, best_metrics, study = tune_xgboost(
        X_train, X_test, y_train, y_test, n_trials=n_optuna_trials
    )

    best_threshold = float(optimize_threshold(best_model, X_test, y_test, beta=1.0))

    y_prob_final = best_model.predict_proba(X_test)[:, 1]
    y_pred_final = (y_prob_final >= best_threshold).astype(int)

    final_metrics = _evaluate(
        y_test, y_pred_final, y_prob_final,
        model_name=f"XGBoost (threshold={best_threshold:.3f})"
    )

    plot_confusion(y_test, y_pred_final, best_threshold)
    plot_feature_importance(best_model, FEATURE_COLS)

    output = {
        'best_threshold': best_threshold,
        'baseline'      : baseline_metrics,
        'xgb_default'   : xgb_metrics,
        'xgb_tuned'     : best_metrics,
        'xgb_final'     : final_metrics,
    }

    with open(MODELS_DIR / 'metrics.json', 'w') as f:
        json.dump(output, f, indent=2)

    logger.info("Training complete.")
    return output


if __name__ == "__main__":
    results = run_training(n_optuna_trials=50)