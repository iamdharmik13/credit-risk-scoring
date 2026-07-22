# ============================================================
#  CREDIT RISK SCORING — LOAN DEFAULT PREDICTION
#  Author : Dharmik Panchal
#  Domain : BFSI — Credit Risk & Lending
#  Tools  : Python, Pandas, Scikit-learn, Matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── STYLE ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3a3d5c",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#b0b0c0",
    "ytick.color":      "#b0b0c0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3e",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

ACCENT  = "#00d4ff"
DANGER  = "#ff4d6d"
WARNING = "#ffd166"
SUCCESS = "#06d6a0"
PURPLE  = "#a78bfa"

# ── LOAD DATA ─────────────────────────────────────────────────
print("=" * 60)
print("  CREDIT RISK SCORING — LOAN DEFAULT PREDICTION")
print("=" * 60)

df = pd.read_csv("borrowers.csv", parse_dates=["application_date"])
df_acc = pd.read_csv("loan_accounts.csv")
df_repay = pd.read_csv("repayment_history.csv")

print(f"\n📊 Dataset Loaded:")
print(f"   Borrowers          : {len(df):,}")
print(f"   Loan Accounts       : {len(df_acc):,}")
print(f"   Repayment Records   : {len(df_repay):,}")
print(f"   Default Rate        : {df['is_default'].mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════
#  SECTION 1 — EDA
# ════════════════════════════════════════════════════════════
print("\n📈 Running EDA...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Credit Risk Scoring — Exploratory Data Analysis",
             fontsize=18, fontweight="bold", color=ACCENT, y=0.98)

# 1a. Default vs Non-default
counts = df["is_default"].value_counts()
axes[0,0].pie(counts, labels=["No Default","Default"],
              colors=[SUCCESS, DANGER], autopct="%1.1f%%", startangle=90,
              textprops={"color":"white","fontsize":11},
              wedgeprops={"edgecolor":"#0f1117","linewidth":2})
axes[0,0].set_title("Loan Default Distribution", fontsize=13, color=ACCENT)

# 1b. Credit score distribution
axes[0,1].hist(df[df["is_default"]==0]["credit_score"], bins=40, color=SUCCESS, alpha=0.7, label="No Default", density=True)
axes[0,1].hist(df[df["is_default"]==1]["credit_score"], bins=40, color=DANGER,  alpha=0.7, label="Default",    density=True)
axes[0,1].set_title("Credit Score Distribution", fontsize=13, color=ACCENT)
axes[0,1].set_xlabel("Credit Score")
axes[0,1].legend()

# 1c. Default rate by employment type
emp_def = df.groupby("employment_type")["is_default"].mean().sort_values() * 100
axes[0,2].barh(emp_def.index, emp_def.values,
               color=[DANGER if v > 18 else WARNING if v > 12 else SUCCESS for v in emp_def.values])
axes[0,2].set_title("Default Rate by Employment Type", fontsize=13, color=ACCENT)
axes[0,2].set_xlabel("Default Rate (%)")

# 1d. DTI ratio vs default
axes[1,0].scatter(df[df["is_default"]==0]["dti_ratio"], df[df["is_default"]==0]["credit_score"],
                  c=SUCCESS, alpha=0.2, s=8, label="No Default")
axes[1,0].scatter(df[df["is_default"]==1]["dti_ratio"], df[df["is_default"]==1]["credit_score"],
                  c=DANGER, alpha=0.4, s=10, label="Default", marker="X")
axes[1,0].set_title("DTI Ratio vs Credit Score", fontsize=13, color=ACCENT)
axes[1,0].set_xlabel("Debt-to-Income Ratio")
axes[1,0].set_ylabel("Credit Score")
axes[1,0].legend()

# 1e. Default rate by loan purpose
purp_def = df.groupby("loan_purpose")["is_default"].mean().sort_values() * 100
axes[1,1].bar(purp_def.index, purp_def.values, color=PURPLE, edgecolor="#0f1117")
axes[1,1].set_title("Default Rate by Loan Purpose", fontsize=13, color=ACCENT)
axes[1,1].set_ylabel("Default Rate (%)")
axes[1,1].tick_params(axis="x", rotation=30)

# 1f. Monthly application trend
monthly = df.groupby(df["application_date"].dt.to_period("M")).agg(
    total=("is_default","count"), defaults=("is_default","sum")).reset_index()
x = range(len(monthly))
axes[1,2].bar(x, monthly["total"], color=SUCCESS, alpha=0.7, label="Total Applications")
axes[1,2].bar(x, monthly["defaults"], color=DANGER, alpha=0.9, label="Defaults")
axes[1,2].set_title("Monthly Applications & Defaults", fontsize=13, color=ACCENT)
axes[1,2].set_xticks(list(x)[::3])
axes[1,2].set_xticklabels([str(monthly["application_date"].iloc[i]) for i in range(0,len(monthly),3)],
                           rotation=45, fontsize=7)
axes[1,2].legend()

plt.tight_layout()
plt.savefig("01_eda_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   ✅ 01_eda_dashboard.png saved")

# ════════════════════════════════════════════════════════════
#  SECTION 2 — RISK SCORING
# ════════════════════════════════════════════════════════════
print("\n🎯 Building Borrower Risk Scores...")

df["risk_score"] = 0
df.loc[df["credit_score"] < 600, "risk_score"] += 30
df.loc[(df["credit_score"] >= 600) & (df["credit_score"] < 700), "risk_score"] += 12
df.loc[df["dti_ratio"] > 0.5, "risk_score"] += 22
df.loc[(df["dti_ratio"] > 0.35) & (df["dti_ratio"] <= 0.5), "risk_score"] += 10
df.loc[df["previous_default"] == 1, "risk_score"] += 25
df.loc[df["existing_loans"] >= 3, "risk_score"] += 8
df.loc[df["monthly_income"] < 30000, "risk_score"] += 10
df.loc[df["years_employed"] < 1, "risk_score"] += 5

df["risk_tier"] = pd.cut(df["risk_score"], bins=[-1, 20, 45, 200], labels=["LOW","MEDIUM","HIGH"])

tier_counts = df["risk_tier"].value_counts()
print(f"\n   Risk Distribution:")
for tier, cnt in tier_counts.items():
    print(f"   {tier:6s} : {cnt} borrowers")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Borrower Risk Scoring & Segmentation", fontsize=16, fontweight="bold", color=ACCENT)

tier_clrs = {"LOW": SUCCESS, "MEDIUM": WARNING, "HIGH": DANGER}
clrs = [tier_clrs[t] for t in tier_counts.index]
axes[0].pie(tier_counts, labels=tier_counts.index, colors=clrs, autopct="%1.1f%%",
            startangle=90, textprops={"color":"white","fontsize":12},
            wedgeprops={"edgecolor":"#0f1117","linewidth":2})
axes[0].set_title("Borrower Risk Distribution", fontsize=13, color=ACCENT)

axes[1].hist(df["risk_score"], bins=30, color=ACCENT, edgecolor="#0f1117")
axes[1].axvline(20, color=SUCCESS, linestyle="--", linewidth=2, label="LOW threshold")
axes[1].axvline(45, color=DANGER,  linestyle="--", linewidth=2, label="HIGH threshold")
axes[1].set_title("Risk Score Distribution", fontsize=13, color=ACCENT)
axes[1].set_xlabel("Risk Score")
axes[1].legend()

city_risk = df[df["risk_tier"]=="HIGH"].groupby("city").size().sort_values(ascending=True)
axes[2].barh(city_risk.index, city_risk.values, color=DANGER, edgecolor="#0f1117")
axes[2].set_title("HIGH Risk Borrowers by City", fontsize=13, color=ACCENT)
axes[2].set_xlabel("Count")

plt.tight_layout()
plt.savefig("02_risk_scoring.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   ✅ 02_risk_scoring.png saved")

# ════════════════════════════════════════════════════════════
#  SECTION 3 — ML MODELS
# ════════════════════════════════════════════════════════════
print("\n🤖 Training ML Models...")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
lr.fit(X_train_s, y_train)
lr_proba = lr.predict_proba(X_test_s)[:,1]
lr_pred  = lr.predict(X_test_s)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
                             class_weight="balanced", min_samples_leaf=5)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:,1]
rf_pred  = rf.predict(X_test)

lr_auc = roc_auc_score(y_test, lr_proba)
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"\n   Logistic Regression AUC : {lr_auc:.4f}")
print(f"   Random Forest       AUC : {rf_auc:.4f}")
print(f"\n   Random Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=["No Default","Default"], digits=3))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("ML Default Prediction Models — Evaluation", fontsize=16, fontweight="bold", color=ACCENT)

cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Default","Default"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"Random Forest Confusion Matrix\nAUC: {rf_auc:.4f}", fontsize=12, color=ACCENT)
axes[0].set_facecolor("#1a1d2e")

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
axes[1].plot(fpr_lr, tpr_lr, color=WARNING, lw=2, label=f"Logistic Reg (AUC={lr_auc:.3f})")
axes[1].plot(fpr_rf, tpr_rf, color=ACCENT,  lw=2, label=f"Random Forest (AUC={rf_auc:.3f})")
axes[1].plot([0,1],[0,1],"k--",alpha=0.4)
axes[1].set_title("ROC Curve Comparison", fontsize=13, color=ACCENT)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(10)
bars = axes[2].barh(fi.index, fi.values, color=PURPLE, edgecolor="#0f1117")
axes[2].set_title("Top 10 Feature Importances", fontsize=13, color=ACCENT)
axes[2].set_xlabel("Importance")
for bar, val in zip(bars, fi.values):
    axes[2].text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", color="white", fontsize=8)

plt.tight_layout()
plt.savefig("03_ml_evaluation.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   ✅ 03_ml_evaluation.png saved")

# ════════════════════════════════════════════════════════════
#  SECTION 4 — REPAYMENT BEHAVIOUR ANALYSIS
# ════════════════════════════════════════════════════════════
print("\n💳 Repayment Behaviour Analysis...")

repay_summary = df_repay.groupby("borrower_id").agg(
    total_emis=("repayment_id","count"),
    late_count=("payment_status", lambda x: (x=="Late").sum()),
    avg_days_late=("days_late","mean")
).reset_index()
repay_summary["late_pct"] = repay_summary["late_count"] / repay_summary["total_emis"] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Repayment Behaviour Analysis", fontsize=16, fontweight="bold", color=ACCENT)

axes[0].hist(repay_summary["late_pct"], bins=20, color=WARNING, edgecolor="#0f1117")
axes[0].set_title("Late Payment % Distribution", fontsize=13, color=ACCENT)
axes[0].set_xlabel("Late Payment %")
axes[0].set_ylabel("Number of Borrowers")

status_counts = df_acc["account_status"].value_counts()
status_colors = {"Active": SUCCESS, "Closed": ACCENT, "NPA": DANGER}
clrs2 = [status_colors.get(s, PURPLE) for s in status_counts.index]
axes[1].pie(status_counts, labels=status_counts.index, colors=clrs2, autopct="%1.1f%%",
            startangle=90, textprops={"color":"white","fontsize":11},
            wedgeprops={"edgecolor":"#0f1117","linewidth":2})
axes[1].set_title("Loan Account Status Distribution", fontsize=13, color=ACCENT)

plt.tight_layout()
plt.savefig("04_repayment_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   ✅ 04_repayment_analysis.png saved")

# Save for Power BI
df[["borrower_id","age","monthly_income","employment_type","credit_score","dti_ratio",
    "loan_amount","loan_purpose","city","previous_default","risk_score","risk_tier",
    "is_default"]].to_csv("borrower_risk_scores.csv", index=False)

# ════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PROJECT SUMMARY")
print("=" * 60)
print(f"  Total Borrowers Analyzed : {len(df):,}")
print(f"  Default Rate             : {df['is_default'].mean()*100:.1f}%")
print(f"  HIGH Risk Borrowers      : {(df['risk_tier']=='HIGH').sum():,}")
print(f"  MEDIUM Risk Borrowers    : {(df['risk_tier']=='MEDIUM').sum():,}")
print(f"  LOW Risk Borrowers       : {(df['risk_tier']=='LOW').sum():,}")
print(f"  Random Forest AUC Score  : {rf_auc:.4f}")
print(f"  Logistic Regression AUC  : {lr_auc:.4f}")
print("\n  Output Files:")
print("  01_eda_dashboard.png")
print("  02_risk_scoring.png")
print("  03_ml_evaluation.png")
print("  04_repayment_analysis.png")
print("  borrower_risk_scores.csv")
print("=" * 60)
