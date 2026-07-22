import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

# ── 1. BORROWERS ──────────────────────────────────────────────
NUM_BORROWERS = 10000

employment_types = ["Salaried", "Self-Employed", "Business Owner", "Government Employee"]
education_levels  = ["High School", "Graduate", "Post Graduate", "Doctorate"]
loan_purposes     = ["Home", "Vehicle", "Education", "Personal", "Business", "Medical"]
marital_status    = ["Single", "Married", "Divorced"]

rows = []
for i in range(1, NUM_BORROWERS + 1):
    age          = random.randint(21, 65)
    income       = round(np.random.lognormal(mean=10.8, sigma=0.5), 2)  # ~ 30k - 3L monthly
    income       = max(15000, min(income, 500000))
    employment   = random.choice(employment_types)
    education    = random.choice(education_levels)
    credit_score = int(np.clip(np.random.normal(680, 90), 300, 900))
    existing_loans = random.randint(0, 4)
    loan_amount  = round(random.uniform(50000, 2000000), 2)
    loan_purpose = random.choice(loan_purposes)
    loan_tenure  = random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240])
    dti_ratio    = round(random.uniform(0.05, 0.75), 2)   # debt-to-income
    marital      = random.choice(marital_status)
    dependents   = random.randint(0, 4)
    city         = random.choice(["Ahmedabad","Mumbai","Delhi","Surat","Pune","Bengaluru","Jaipur"])
    years_employed = round(random.uniform(0, 30), 1)
    prev_default = random.choice([0, 0, 0, 0, 1])  # 20% prior default

    # ── Default probability logic (synthetic but realistic) ──
    risk_score = 0
    if credit_score < 600: risk_score += 28
    elif credit_score < 700: risk_score += 10
    if dti_ratio > 0.5: risk_score += 20
    elif dti_ratio > 0.35: risk_score += 7
    if prev_default == 1: risk_score += 25
    if existing_loans >= 3: risk_score += 8
    if income < 30000: risk_score += 10
    if years_employed < 1: risk_score += 7

    default_prob = min(risk_score / 100, 0.85) * 0.55  # scale down to realistic ~15-18%
    is_default = int(random.random() < default_prob)

    rows.append({
        "borrower_id": f"BOR{i:05d}",
        "age": age,
        "monthly_income": income,
        "employment_type": employment,
        "years_employed": years_employed,
        "education": education,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "loan_tenure_months": loan_tenure,
        "dti_ratio": dti_ratio,
        "marital_status": marital,
        "dependents": dependents,
        "city": city,
        "previous_default": prev_default,
        "application_date": fake.date_between(start_date="-2y", end_date="today"),
        "is_default": is_default
    })

df = pd.DataFrame(rows)

# ── 2. LOAN ACCOUNTS (post-approval tracking) ───────────────────
df_loans = df[df["is_default"] == 0].copy()
approved = df_loans.sample(frac=0.85, random_state=1)  # not all approved get tracked records
loan_accounts = []
for i, row in enumerate(approved.itertuples(), 1):
    loan_accounts.append({
        "account_id": f"LA{i:05d}",
        "borrower_id": row.borrower_id,
        "branch": random.choice(["Ahmedabad Main","Surat Branch","Mumbai HQ","Delhi North","Pune East"]),
        "disbursed_amount": row.loan_amount,
        "emi_amount": round(row.loan_amount * 1.12 / row.loan_tenure_months, 2),
        "account_status": random.choice(["Active","Active","Active","Closed","NPA"])
    })
df_accounts = pd.DataFrame(loan_accounts)

# ── 3. REPAYMENT HISTORY ────────────────────────────────────────
repayments = []
rid = 1
for row in df_accounts.sample(min(2000, len(df_accounts)), random_state=2).itertuples():
    n_months = random.randint(3, 12)
    for m in range(n_months):
        late = random.random() < (0.25 if row.account_status == "NPA" else 0.05)
        repayments.append({
            "repayment_id": f"RPY{rid:06d}",
            "account_id": row.account_id,
            "borrower_id": row.borrower_id,
            "month_number": m + 1,
            "emi_due": row.emi_amount,
            "payment_status": "Late" if late else "On-Time",
            "days_late": random.randint(5, 60) if late else 0
        })
        rid += 1
df_repay = pd.DataFrame(repayments)

# ── SAVE ──────────────────────────────────────────────────────
df.to_csv("borrowers.csv", index=False)
df_accounts.to_csv("loan_accounts.csv", index=False)
df_repay.to_csv("repayment_history.csv", index=False)

print("✅ Credit Risk Dataset Generated!")
print(f"   Borrowers         : {len(df):,}")
print(f"   Loan Accounts      : {len(df_accounts):,}")
print(f"   Repayment Records  : {len(df_repay):,}")
print(f"   Default Rate       : {df['is_default'].mean()*100:.1f}%")
