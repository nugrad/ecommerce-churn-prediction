"""
Streamlit Dashboard — E-Commerce Churn Prediction

Three pages:
    1. Overview      → project summary, model metrics, SHAP rankings
    2. Single        → interactive single customer predictor
    3. Batch         → score all customers, download results
"""

import streamlit         as st
import pandas            as pd
import numpy             as np
import joblib
import json
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Churn Prediction Dashboard",
    page_icon  = "🛒",
    layout     = "wide",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load(ROOT / "models" / "xgb_tuned.pkl")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return model, metrics

@st.cache_data
def load_feature_matrix():
    return pd.read_csv(ROOT / "data" / "processed" / "feature_matrix.csv")

model, metrics = load_model()
THRESHOLD = metrics['best_threshold']

FEATURE_COLS = [
    'delivery_delay_days', 'approval_delay_hours',
    'was_late', 'delivery_speed_days', 'estimated_speed_days',
    'speed_vs_promise_ratio', 'has_review', 'review_score_filled',
    'low_review', 'high_review', 'purchase_hour', 'purchase_dayofweek',
    'purchase_month', 'is_weekend', 'is_business_hours',
    'state_churn_rate', 'is_sao_paulo', 'is_remote_state', 'slow_approval'
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def assign_risk_tier(prob: float) -> str:
    if prob >= 0.20:   return "LIKELY_TO_RETURN"
    elif prob >= 0.10: return "UNCERTAIN"
    else:              return "AT_RISK"

def tier_color(tier: str) -> str:
    return {
        "LIKELY_TO_RETURN": "🟢",
        "UNCERTAIN"       : "🟡",
        "AT_RISK"         : "🔴"
    }.get(tier, "⚪")

def tier_hex(tier: str) -> str:
    return {
        "LIKELY_TO_RETURN": "#27ae60",
        "UNCERTAIN"       : "#f39c12",
        "AT_RISK"         : "#e74c3c"
    }.get(tier, "#95a5a6")

def predict(features: dict) -> tuple:
    X    = pd.DataFrame([features])[FEATURE_COLS]
    prob = float(model.predict_proba(X)[0, 1])
    tier = assign_risk_tier(prob)
    return prob, prob >= THRESHOLD, tier

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    X                        = df[FEATURE_COLS]
    probs                    = model.predict_proba(X)[:, 1]
    out                      = df.copy()
    out['return_probability'] = np.round(probs, 4)
    out['risk_tier']          = [assign_risk_tier(p) for p in probs]
    out['will_return']        = probs >= THRESHOLD
    return out.sort_values('return_probability', ascending=False).reset_index(drop=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://img.icons8.com/fluency/96/shopping-cart.png",
    width=60
)
st.sidebar.title("Churn Predictor")
st.sidebar.markdown("**E-Commerce Repeat Purchase Prediction**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "👤 Single Customer", "📦 Batch Scoring"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.metric("AUC-ROC",   f"{metrics['xgb_tuned']['auc_roc']:.4f}")
st.sidebar.metric("AUC-PR",    f"{metrics['xgb_tuned']['auc_pr']:.4f}")
st.sidebar.metric("Threshold", f"{THRESHOLD:.4f}")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with XGBoost + Optuna  \n"
    "Dataset: Olist Brazil E-Commerce  \n"
    "93,350 customers · 19 features"
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🛒 E-Commerce Churn Prediction")
    st.markdown(
        "Predicts whether a **first-time customer will make a second purchase**, "
        "using behavioral signals from their first order."
    )

    # ── Key numbers ──
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",   "93,350")
    c2.metric("Repeat Rate",       "3.0%")
    c3.metric("Features Built",    "19")
    c4.metric("Model",             "XGBoost")

    # ── Problem framing ──
    st.markdown("---")
    st.subheader("Why Repeat Purchase Prediction?")
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Original approach (abandoned):**  \n"
            "Snapshot-based churn definition produced **99.3% churn rate** "
            "— mathematically unlearnable. Any model predicting everyone churns "
            "scores 99.3% accuracy while being completely useless."
        )
    with col2:
        st.success(
            "**Revised approach (used):**  \n"
            "Repeat purchase prediction — did the customer ever buy again?  \n"
            "Produces a **3% minority class**, learnable with XGBoost's "
            "`scale_pos_weight`. This is the correct framing for a "
            "non-subscription marketplace."
        )

    # ── Model comparison ──
    st.markdown("---")
    st.subheader("Model Comparison")

    comparison = pd.DataFrame([
        {
            "Model"         : "Baseline — Logistic Regression",
            "AUC-ROC"       : metrics['baseline']['auc_roc'],
            "AUC-PR"        : metrics['baseline']['auc_pr'],
            "F1 (minority)" : metrics['baseline']['f1_min'],
        },
        {
            "Model"         : "XGBoost (default params)",
            "AUC-ROC"       : metrics['xgb_default']['auc_roc'],
            "AUC-PR"        : metrics['xgb_default']['auc_pr'],
            "F1 (minority)" : metrics['xgb_default']['f1_min'],
        },
        {
            "Model"         : "XGBoost (Optuna tuned) ✓",
            "AUC-ROC"       : metrics['xgb_tuned']['auc_roc'],
            "AUC-PR"        : metrics['xgb_tuned']['auc_pr'],
            "F1 (minority)" : metrics['xgb_tuned']['f1_min'],
        },
    ])

    st.dataframe(
        comparison.style
        .highlight_max(subset=["AUC-ROC", "AUC-PR", "F1 (minority)"], color="#d4edda")
        .format({"AUC-ROC": "{:.4f}", "AUC-PR": "{:.4f}", "F1 (minority)": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    # ── SHAP feature importance ──
    st.markdown("---")
    st.subheader("Feature Importance (SHAP)")

    shap_df = pd.DataFrame({
        "Feature"        : [
            "state_churn_rate", "purchase_month", "estimated_speed_days",
            "approval_delay_hours", "purchase_hour", "delivery_speed_days",
            "delivery_delay_days", "review_score_filled",
            "speed_vs_promise_ratio", "is_sao_paulo"
        ],
        "Mean |SHAP|"    : [
            0.0867, 0.0821, 0.0756, 0.0695, 0.0465,
            0.0364, 0.0248, 0.0204, 0.0160, 0.0094
        ],
        "Business Meaning": [
            "State logistics quality proxy",
            "Seasonality — holiday vs regular buyers",
            "Delivery promise length",
            "Merchant responsiveness",
            "Time-of-day purchase behavior",
            "Actual delivery experience",
            "Lateness vs promised date",
            "Post-purchase satisfaction",
            "Speed vs expectation gap",
            "São Paulo market behavior"
        ]
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        shap_df['Feature'][::-1],
        shap_df['Mean |SHAP|'][::-1],
        color='#3498db', edgecolor='white'
    )
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title('Top 10 Features by SHAP Importance', fontweight='bold', fontsize=13)
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # ── Honest limitations ──
    st.markdown("---")
    st.subheader("Honest Model Limitations")
    st.warning(
        "**AUC-ROC of 0.58 reflects a genuine data constraint, not a modeling failure.**  \n\n"
        "First-order behavioral features explain only a fraction of repeat purchase intent. "
        "Missing high-signal features include: product category, order value, seller rating, "
        "customer browsing behavior, and discount usage.  \n\n"
        "The model delivers **+30% lift over random** on AUC-PR and is suitable for "
        "prioritizing retention outreach — not as a sole decision system."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE CUSTOMER PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👤 Single Customer":
    st.title("👤 Single Customer Prediction")
    st.markdown(
        "Enter details from a customer's first order to predict "
        "their likelihood of making a second purchase."
    )

    st.markdown("---")

    with st.form("predict_form"):

        # ── Delivery ──
        st.subheader("📦 Delivery Experience")
        col1, col2, col3 = st.columns(3)

        delivery_delay  = col1.number_input(
            "Delivery Delay (days)",
            value=0.0, step=0.5,
            help="Actual delivery minus estimated. Negative means arrived early."
        )
        actual_speed    = col2.number_input(
            "Actual Delivery Speed (days)",
            min_value=0.0, value=7.0, step=0.5,
            help="Days from purchase to delivery"
        )
        promised_speed  = col3.number_input(
            "Promised Delivery Speed (days)",
            min_value=0.0, value=7.0, step=0.5,
            help="Days from purchase to estimated delivery date"
        )

        # ── Merchant ──
        st.subheader("🏪 Merchant Behavior")
        col4, col5 = st.columns(2)
        approval_hours = col4.number_input(
            "Approval Delay (hours)",
            min_value=0.0, value=2.0, step=0.5,
            help="Hours from purchase to order approval"
        )

        # ── Review ──
        st.subheader("⭐ Customer Review")
        col6, col7 = st.columns(2)
        left_review  = col6.selectbox(
            "Did customer leave a review?",
            ["Yes", "No"],
            index=0
        )
        review_score = col7.slider(
            "Review Score",
            min_value=1.0, max_value=5.0, value=4.0, step=0.5,
            help="Set to 3.0 if no review was left"
        )
        if left_review == "No":
            review_score = 3.0
            st.caption("Review score set to 3.0 (neutral) — no review given.")

        # ── Timing ──
        st.subheader("⏰ Purchase Timing")
        col8, col9, col10 = st.columns(3)
        purchase_month = col8.selectbox(
            "Month of Purchase",
            options=list(range(1, 13)),
            format_func=lambda x: [
                "Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"
            ][x-1],
            index=5
        )
        purchase_hour  = col9.slider("Hour of Purchase", 0, 23, 12)
        purchase_dow   = col10.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
            index=2
        )

        # ── Geography ──
        st.subheader("📍 Customer Location")
        col11, col12, col13 = st.columns(3)
        state_return_rate = col11.slider(
            "State Return Rate",
            min_value=0.80, max_value=1.0, value=0.97, step=0.01,
            help="Lower = better logistics. Most states are 0.95-0.99."
        )
        is_sp      = col12.selectbox("Is customer in São Paulo?", ["No", "Yes"])
        is_remote  = col13.selectbox("Is customer in remote state?", ["No", "Yes"])

        submitted = st.form_submit_button(
            "🔍 Predict Return Likelihood",
            use_container_width=True
        )

    # ── On submit ──
    if submitted:
        was_late          = int(delivery_delay > 0)
        slow_approval     = int(approval_hours > 24)
        has_review        = int(left_review == "Yes")
        low_review        = int(review_score <= 2)
        high_review       = int(review_score >= 4)
        is_weekend        = int(purchase_dow >= 5)
        is_biz_hours      = int(9 <= purchase_hour <= 18 and purchase_dow < 5)
        speed_ratio       = round(
            promised_speed / actual_speed if actual_speed > 0 else 1.0, 4
        )

        features = {
            'delivery_delay_days'   : delivery_delay,
            'approval_delay_hours'  : approval_hours,
            'was_late'              : was_late,
            'delivery_speed_days'   : actual_speed,
            'estimated_speed_days'  : promised_speed,
            'speed_vs_promise_ratio': speed_ratio,
            'has_review'            : has_review,
            'review_score_filled'   : review_score,
            'low_review'            : low_review,
            'high_review'           : high_review,
            'purchase_hour'         : purchase_hour,
            'purchase_dayofweek'    : purchase_dow,
            'purchase_month'        : purchase_month,
            'is_weekend'            : is_weekend,
            'is_business_hours'     : is_biz_hours,
            'state_churn_rate'      : state_return_rate,
            'is_sao_paulo'          : int(is_sp == "Yes"),
            'is_remote_state'       : int(is_remote == "Yes"),
            'slow_approval'         : slow_approval,
        }

        prob, will_return, tier = predict(features)

        # ── Result display ──
        st.markdown("---")
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Return Probability", f"{prob:.1%}")
        r2.metric("Prediction", "Will Return ✅" if will_return else "Won't Return ❌")
        r3.metric("Risk Tier", f"{tier_color(tier)} {tier}")

        # Probability bar
        fig, ax = plt.subplots(figsize=(9, 1.5))
        ax.barh([""], [1.0], color='#ecf0f1', height=0.5)
        ax.barh([""], [prob], color=tier_hex(tier), height=0.5)
        ax.axvline(
            THRESHOLD, color='#2c3e50',
            linestyle='--', linewidth=2,
            label=f'Decision threshold ({THRESHOLD:.2f})'
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Return Probability")
        ax.legend(loc='upper right', fontsize=9)
        ax.spines[['top','right','left']].set_visible(False)
        ax.set_title(
            f"Customer probability: {prob:.1%} vs threshold: {THRESHOLD:.1%}",
            fontweight='bold'
        )
        st.pyplot(fig)
        plt.close()

        # Action box
        st.markdown("---")
        st.subheader("Recommended Business Action")

        if tier == "LIKELY_TO_RETURN":
            st.success(
                "🟢 **LIKELY TO RETURN**  \n\n"
                "Customer shows strong return signal.  \n"
                "→ Enroll in loyalty program  \n"
                "→ Send personalized product recommendations  \n"
                "→ No discount needed — don't erode margin unnecessarily"
            )
        elif tier == "UNCERTAIN":
            st.warning(
                "🟡 **UNCERTAIN**  \n\n"
                "Customer is on the fence.  \n"
                "→ Send follow-up satisfaction email within 48hrs of delivery  \n"
                "→ Offer 10% discount on next purchase  \n"
                "→ Highlight relevant product categories"
            )
        else:
            st.error(
                "🔴 **AT RISK — High Churn Probability**  \n\n"
                "Customer unlikely to return without intervention.  \n"
                "→ Send immediate retention offer (15% off + free shipping)  \n"
                "→ Trigger within 24hrs of delivery  \n"
                "→ Personalize based on product category purchased"
            )

        # Feature summary
        with st.expander("View features sent to model"):
            st.json(features)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH SCORING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📦 Batch Scoring":
    st.title("📦 Batch Customer Scoring")
    st.markdown("Score all customers at once and download results for your CRM.")

    tab1, tab2 = st.tabs(["Score Existing Data", "Upload CSV"])

    # ── Tab 1: Score existing data ────────────────────────────────────────────
    with tab1:
        st.markdown(
            "Scores all 93,350 customers from the processed feature matrix. "
            "Results sorted by return probability — highest value customers first."
        )

        if st.button("▶ Run Batch Scoring", use_container_width=True):
            with st.spinner("Scoring all customers... (this takes ~10 seconds)"):
                df     = load_feature_matrix()
                scored = score_dataframe(df)

            st.success(f"Scored {len(scored):,} customers successfully.")
            st.markdown("---")

            # ── Summary metrics ──
            likely   = (scored['risk_tier'] == 'LIKELY_TO_RETURN').sum()
            uncertain= (scored['risk_tier'] == 'UNCERTAIN').sum()
            at_risk  = (scored['risk_tier'] == 'AT_RISK').sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Scored",      f"{len(scored):,}")
            m2.metric("🟢 Likely Return",  f"{likely:,}")
            m3.metric("🟡 Uncertain",      f"{uncertain:,}")
            m4.metric("🔴 At Risk",        f"{at_risk:,}")

            # ── Tier distribution chart ──
            st.markdown("---")
            st.subheader("Risk Tier Distribution")

            tier_order  = ['LIKELY_TO_RETURN', 'UNCERTAIN', 'AT_RISK']
            tier_counts = scored['risk_tier'].value_counts().reindex(tier_order)
            tier_colors = ['#27ae60', '#f39c12', '#e74c3c']

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Bar chart
            bars = axes[0].bar(
                ['Likely\nReturn', 'Uncertain', 'At Risk'],
                tier_counts.values,
                color=tier_colors,
                edgecolor='white', linewidth=1.5
            )
            for bar, val in zip(bars, tier_counts.values):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 200,
                    f'{val:,}', ha='center', fontweight='bold', fontsize=10
                )
            axes[0].set_ylabel('Customers')
            axes[0].set_title('Customer Count by Risk Tier', fontweight='bold')
            axes[0].spines[['top', 'right']].set_visible(False)

            # Pie chart
            axes[1].pie(
                tier_counts.values,
                labels=['Likely Return', 'Uncertain', 'At Risk'],
                colors=tier_colors,
                autopct='%1.1f%%',
                startangle=90
            )
            axes[1].set_title('Risk Tier Breakdown', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Return probability histogram ──
            st.markdown("---")
            st.subheader("Return Probability Distribution")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(
                scored['return_probability'],
                bins=50, color='#3498db',
                edgecolor='white', linewidth=0.5
            )
            ax.axvline(
                THRESHOLD, color='#e74c3c',
                linestyle='--', linewidth=2,
                label=f'Threshold ({THRESHOLD:.3f})'
            )
            ax.set_xlabel('Return Probability')
            ax.set_ylabel('Number of Customers')
            ax.set_title('Distribution of Return Probabilities', fontweight='bold')
            ax.legend()
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig)
            plt.close()

            # ── Top customers table ──
            st.markdown("---")
            st.subheader("Top 20 — Highest Return Probability")

            display_cols = [
                'customer_unique_id', 'customer_state',
                'return_probability', 'risk_tier',
                'review_score_filled', 'delivery_delay_days', 'purchase_month'
            ]
            available = [c for c in display_cols if c in scored.columns]
            st.dataframe(
                scored.head(20)[available],
                use_container_width=True,
                hide_index=True
            )

            # ── Download ──
            st.markdown("---")
            csv = scored[available].to_csv(index=False)
            st.download_button(
                label    = "⬇ Download Full Scored CSV",
                data     = csv,
                file_name= "batch_scores.csv",
                mime     = "text/csv",
                use_container_width=True
            )

    # ── Tab 2: Upload CSV ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("Upload your own customer CSV with the required feature columns.")

        # Template download
        template = pd.DataFrame(columns=['customer_unique_id'] + FEATURE_COLS)
        st.download_button(
            "⬇ Download Template CSV",
            data      = template.to_csv(index=False),
            file_name = "churn_template.csv",
            mime      = "text/csv"
        )

        st.markdown("Fill in the template and upload it below:")
        uploaded = st.file_uploader("Upload CSV", type="csv")

        if uploaded:
            df      = pd.read_csv(uploaded)
            missing = [c for c in FEATURE_COLS if c not in df.columns]

            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()

            with st.spinner("Scoring uploaded customers..."):
                scored = score_dataframe(df)

            st.success(f"Scored {len(scored):,} customers.")

            st.dataframe(
                scored[[
                    c for c in
                    ['customer_unique_id', 'return_probability', 'risk_tier', 'will_return']
                    + FEATURE_COLS
                    if c in scored.columns
                ]].head(50),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label    = "⬇ Download Scored Results",
                data     = scored.to_csv(index=False),
                file_name= "uploaded_scores.csv",
                mime     = "text/csv",
                use_container_width=True
            )