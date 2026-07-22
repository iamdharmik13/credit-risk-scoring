# 💳 Credit Risk Scoring — Loan Default Prediction
### BFSI-grade Credit Scoring System with Interactive UI

> **Domain:** BFSI — Credit Risk & Lending  
> **Tools:** Python · MySQL · Power BI · Scikit-learn · Streamlit · HTML/JS  
> **Author:** Dharmik Panchal

---

## 📌 Project Overview

An end-to-end credit risk scoring system that predicts loan default probability and segments borrowers into **LOW / MEDIUM / HIGH** risk tiers. Built for lenders to make faster, data-driven approval decisions — includes two interactive interfaces (Streamlit + standalone HTML) on top of the core ML pipeline.

---

## 🖥️ Interactive Interfaces (New!)

| Interface | Description |
|-----------|-------------|
| **`app.py`** (Streamlit) | Full web app — risk calculator, portfolio overview, model insights across 3 tabs |
| **`dashboard.html`** | Standalone HTML page — real logistic regression coefficients embedded in JavaScript, works with just a double-click, no server needed |

Both let you enter a borrower's details and get an instant risk verdict powered by the actual trained model.

---

## 🗃️ Dataset

| Table | Records | Description |
|-------|---------|-------------|
| `borrowers.csv` | 10,000 | Demographics, income, credit score, loan details |
| `loan_accounts.csv` | ~7,100 | Disbursed loans, EMI, account status |
| `repayment_history.csv` | ~15,000 | Month-wise EMI payment tracking |

---

## 🗄️ MySQL — 8 Credit Risk Queries

| Query | Purpose |
|-------|---------|
| High-Risk Borrower Identification | Low credit score + high DTI + prior default |
| Default Rate by Employment Type | Salaried vs Self-Employed vs Business Owner |
| Credit Score Band Segmentation | CTE + RANK() window function |
| Late Payment Pattern Detection | Borrowers with repeated late EMIs |
| Branch-wise NPA Report | Non-performing asset rate by branch |
| Loan Purpose Risk Analysis | Home vs Vehicle vs Personal default rates |
| DTI Ratio vs Default Correlation | Bucketed CASE + GROUP BY analysis |
| Monthly Application & Default Trend | Time-series risk tracking |

---

## 🤖 Machine Learning

**Models:** Logistic Regression & Random Forest (class-balanced for imbalanced default data)

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | 0.674 |
| Random Forest | 0.684 |

**Key Features:** Credit score, DTI ratio, previous default history, existing loans, employment type, income — identified via feature importance analysis.

---

## 📊 Visualizations

| File | Content |
|------|---------|
| `01_eda_dashboard.png` | Default distribution, credit score analysis, employment risk, DTI scatter |
| `02_risk_scoring.png` | Risk tier segmentation, risk score distribution, city-wise risk |
| `03_ml_evaluation.png` | Confusion matrix, ROC curves, feature importance |
| `04_repayment_analysis.png` | Late payment patterns, loan account status |
| `powerbi_dashboard.png` | Power BI compliance dashboard |

---

## 📁 Project Structure

```
credit-risk-scoring/
│
├── generate_data.py              # Synthetic dataset generation
├── credit_risk_analysis.py       # Main Python ML + EDA script
├── credit_schema_and_queries.sql # MySQL schema + 8 risk queries
├── train_model_for_app.py        # Trains & saves model for Streamlit
├── app.py                        # Streamlit interactive web app
├── dashboard.html                # Standalone HTML risk calculator
├── model_coefficients.json       # Exported LR coefficients (used in HTML)
│
├── borrowers.csv
├── loan_accounts.csv
├── repayment_history.csv
├── borrower_risk_scores.csv      # Output for Power BI
│
├── 01_eda_dashboard.png
├── 02_risk_scoring.png
├── 03_ml_evaluation.png
├── 04_repayment_analysis.png
└── powerbi_dashboard.png
```

---

## ⚙️ How to Run

```bash
pip install pandas numpy scikit-learn matplotlib faker streamlit joblib

# 1. Generate dataset
python generate_data.py

# 2. Run full ML analysis
python credit_risk_analysis.py

# 3. Train model for the interactive app
python train_model_for_app.py

# 4. Launch Streamlit app
streamlit run app.py

# 5. Or just open dashboard.html directly in any browser — no server needed
```

---

## 💡 Key Insights

- **16.0%** overall default rate across 10,000 borrowers
- **HIGH risk tier**: ~2,000 borrowers (20%) needing manual underwriting review
- **Home loans** show the highest default rate (18.1%) among all loan purposes
- **Credit score** and **DTI ratio** are the strongest predictors of default
- **Previous default history** roughly doubles a borrower's risk contribution

---

## 🔗 Technologies Used

`Python` `MySQL` `Power BI` `Scikit-learn` `Streamlit` `Pandas` `NumPy` `Matplotlib` `HTML/CSS/JS`

---

## 📬 Contact

**Dharmik Panchal**
📧 iamdharmik13@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/iamdharmik1334) · [GitHub](https://github.com/iamdharmik13)
