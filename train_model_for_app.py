# ============================================================
#  Train & Save Model for Streamlit App
# ============================================================
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("borrowers.csv")

le_emp = LabelEncoder(); le_edu = LabelEncoder(); le_purp = LabelEncoder()
le_mar = LabelEncoder(); le_city = LabelEncoder()

df["emp_enc"]  = le_emp.fit_transform(df["employment_type"])
df["edu_enc"]  = le_edu.fit_transform(df["education"])
df["purp_enc"] = le_purp.fit_transform(df["loan_purpose"])
df["mar_enc"]  = le_mar.fit_transform(df["marital_status"])
df["city_enc"] = le_city.fit_transform(df["city"])

FEATURES = ["age","monthly_income","years_employed","credit_score","existing_loans",
            "loan_amount","loan_tenure_months","dti_ratio","dependents","previous_default",
            "emp_enc","edu_enc","purp_enc","mar_enc","city_enc"]

X = df[FEATURES]
y = df["is_default"]

scaler = StandardScaler()
scaler.fit(X)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                n_jobs=-1, class_weight="balanced", min_samples_leaf=5)
model.fit(X, y)

joblib.dump(model, "credit_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump({
    "employment_type": le_emp,
    "education": le_edu,
    "loan_purpose": le_purp,
    "marital_status": le_mar,
    "city": le_city
}, "encoders.pkl")

print("✅ Model, scaler, and encoders saved for Streamlit app!")
