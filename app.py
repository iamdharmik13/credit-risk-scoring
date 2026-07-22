# ============================================================
#  CREDIT RISK SCORING — Interactive Streamlit App
#  Author: Dharmik Panchal
#  Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="wide")

# ── LOAD MODEL & DATA ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("credit_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, scaler, encoders

@st.cache_data
def load_data():
    return pd.read_csv("borrower_risk_scores.csv")

model, scaler, encoders = load_model()
df = load_data()

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800; color: #1F4E79;
        margin-bottom: 0px;
    }
    .sub-header { color: #666; font-size: 1rem; margin-bottom: 20px; }
    .risk-box {
        padding: 20px; border-radius: 12px; text-align: center;
        font-size: 1.4rem; font-weight: 700; color: white; margin-top: 10px;
    }
    .low    { background: linear-gradient(135deg, #06d6a0, #04a878); }
    .medium { background: linear-gradient(135deg, #ffd166, #e0a800); color:#333; }
    .high   { background: linear-gradient(135deg, #ff4d6d, #d61c3f); }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💳 Credit Risk Scoring System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">BFSI-grade Loan Default Prediction | Python · Scikit-learn · MySQL · Power BI</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Risk Calculator", "📊 Portfolio Overview", "📈 Model Insights"])

# ════════════════════════════════════════════════════════════
#  TAB 1 — INTERACTIVE RISK CALCULATOR
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Borrower Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 21, 65, 32)
        monthly_income = st.number_input("Monthly Income (₹)", 15000, 500000, 45000, step=1000)
        years_employed = st.slider("Years Employed", 0.0, 30.0, 3.0, step=0.5)
        credit_score = st.slider("Credit Score", 300, 900, 680)
        existing_loans = st.slider("Existing Loans", 0, 4, 1)

    with col2:
        loan_amount = st.number_input("Loan Amount (₹)", 50000, 2000000, 500000, step=10000)
        loan_tenure = st.selectbox("Loan Tenure (months)", [12,24,36,48,60,84,120,180,240], index=4)
        dti_ratio = st.slider("Debt-to-Income Ratio", 0.05, 0.75, 0.30, step=0.01)
        dependents = st.slider("Dependents", 0, 4, 1)
        previous_default = st.selectbox("Previous Default?", ["No", "Yes"])

    with col3:
        employment_type = st.selectbox("Employment Type", encoders["employment_type"].classes_)
        education = st.selectbox("Education", encoders["education"].classes_)
        loan_purpose = st.selectbox("Loan Purpose", encoders["loan_purpose"].classes_)
        marital_status = st.selectbox("Marital Status", encoders["marital_status"].classes_)
        city = st.selectbox("City", encoders["city"].classes_)

    if st.button("🔎 Calculate Risk", type="primary", use_container_width=True):

        # Build feature vector
        input_data = pd.DataFrame([{
            "age": age, "monthly_income": monthly_income, "years_employed": years_employed,
            "credit_score": credit_score, "existing_loans": existing_loans,
            "loan_amount": loan_amount, "loan_tenure_months": loan_tenure,
            "dti_ratio": dti_ratio, "dependents": dependents,
            "previous_default": 1 if previous_default == "Yes" else 0,
            "emp_enc": encoders["employment_type"].transform([employment_type])[0],
            "edu_enc": encoders["education"].transform([education])[0],
            "purp_enc": encoders["loan_purpose"].transform([loan_purpose])[0],
            "mar_enc": encoders["marital_status"].transform([marital_status])[0],
            "city_enc": encoders["city"].transform([city])[0],
        }])

        default_prob = model.predict_proba(input_data)[0][1]

        # Risk score (rule-based, same formula as bulk analysis)
        risk_score = 0
        if credit_score < 600: risk_score += 30
        elif credit_score < 700: risk_score += 12
        if dti_ratio > 0.5: risk_score += 22
        elif dti_ratio > 0.35: risk_score += 10
        if previous_default == "Yes": risk_score += 25
        if existing_loans >= 3: risk_score += 8
        if monthly_income < 30000: risk_score += 10
        if years_employed < 1: risk_score += 5

        if risk_score <= 20:
            tier, css_class = "LOW RISK", "low"
        elif risk_score <= 45:
            tier, css_class = "MEDIUM RISK", "medium"
        else:
            tier, css_class = "HIGH RISK", "high"

        st.markdown(f"""
        <div class="risk-box {css_class}">
            {tier} — Default Probability: {default_prob*100:.1f}% | Risk Score: {risk_score}/100
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Default Probability", f"{default_prob*100:.1f}%")
        c2.metric("Risk Score", f"{risk_score}/100")
        c3.metric("Recommended EMI", f"₹{round(loan_amount*1.12/loan_tenure):,}")

        with st.expander("📋 See Risk Factor Breakdown"):
            factors = []
            if credit_score < 600: factors.append(("Low Credit Score (<600)", 30))
            elif credit_score < 700: factors.append(("Fair Credit Score (600-700)", 12))
            if dti_ratio > 0.5: factors.append(("High DTI Ratio (>50%)", 22))
            elif dti_ratio > 0.35: factors.append(("Moderate DTI Ratio (35-50%)", 10))
            if previous_default == "Yes": factors.append(("Previous Default History", 25))
            if existing_loans >= 3: factors.append(("Multiple Existing Loans (3+)", 8))
            if monthly_income < 30000: factors.append(("Low Monthly Income (<₹30k)", 10))
            if years_employed < 1: factors.append(("Less than 1 Year Employment", 5))

            if factors:
                for f, pts in factors:
                    st.write(f"🔸 **{f}** → +{pts} points")
            else:
                st.write("✅ No significant risk factors detected — clean profile!")

# ════════════════════════════════════════════════════════════
#  TAB 2 — PORTFOLIO OVERVIEW
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Loan Portfolio Overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Borrowers", f"{len(df):,}")
    k2.metric("Default Rate", f"{df['is_default'].mean()*100:.1f}%")
    k3.metric("HIGH Risk", f"{(df['risk_tier']=='HIGH').sum():,}")
    k4.metric("Avg Loan Amount", f"₹{df['loan_amount'].mean():,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Risk Tier Distribution**")
        tier_counts = df["risk_tier"].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        colors = {"LOW":"#06d6a0","MEDIUM":"#ffd166","HIGH":"#ff4d6d"}
        ax.pie(tier_counts, labels=tier_counts.index, autopct="%1.1f%%",
               colors=[colors[t] for t in tier_counts.index])
        st.pyplot(fig)

    with col2:
        st.write("**Default Rate by City**")
        city_def = df.groupby("city")["is_default"].mean().sort_values() * 100
        fig, ax = plt.subplots(figsize=(5,4))
        ax.barh(city_def.index, city_def.values, color="#a78bfa")
        ax.set_xlabel("Default Rate (%)")
        st.pyplot(fig)

    st.write("**Borrower Data Sample**")
    st.dataframe(df.head(20), use_container_width=True)

# ════════════════════════════════════════════════════════════
#  TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance & Insights")
    st.image("03_ml_evaluation.png", use_container_width=True)
    st.image("01_eda_dashboard.png", use_container_width=True)

st.markdown("---")
st.caption("Developed by Dharmik Panchal | Credit Risk Scoring System | BFSI Domain Project")
