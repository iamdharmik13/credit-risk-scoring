-- ============================================================
--  CREDIT RISK SCORING — LOAN DEFAULT PREDICTION — MySQL
--  Project by: Dharmik Panchal
--  Domain: BFSI — Credit Risk & Lending
-- ============================================================

CREATE DATABASE IF NOT EXISTS credit_risk;
USE credit_risk;

-- ── TABLES ───────────────────────────────────────────────────

CREATE TABLE borrowers (
    borrower_id         VARCHAR(10) PRIMARY KEY,
    age                 INT,
    monthly_income      DECIMAL(12,2),
    employment_type     VARCHAR(30),
    years_employed      DECIMAL(5,1),
    education           VARCHAR(30),
    credit_score        INT,
    existing_loans      INT,
    loan_amount         DECIMAL(12,2),
    loan_purpose        VARCHAR(30),
    loan_tenure_months  INT,
    dti_ratio           DECIMAL(5,2),
    marital_status      VARCHAR(20),
    dependents          INT,
    city                VARCHAR(50),
    previous_default    TINYINT(1),
    application_date    DATE,
    is_default          TINYINT(1)
);
select * from borrowers;
CREATE TABLE loan_accounts (
    account_id          VARCHAR(10) PRIMARY KEY,
    borrower_id          VARCHAR(10),
    branch               VARCHAR(50),
    disbursed_amount      DECIMAL(12,2),
    emi_amount            DECIMAL(12,2),
    account_status        VARCHAR(20),
    FOREIGN KEY (borrower_id) REFERENCES borrowers(borrower_id)
);
select * from loan_accounts;
CREATE TABLE repayment_history (
    repayment_id   VARCHAR(12) PRIMARY KEY,
    account_id     VARCHAR(10),
    borrower_id    VARCHAR(10),
    month_number   INT,
    emi_due        DECIMAL(12,2),
    payment_status VARCHAR(20),
    days_late      INT,
    FOREIGN KEY (account_id)   REFERENCES loan_accounts(account_id),
    FOREIGN KEY (borrower_id)  REFERENCES borrowers(borrower_id)
);
select * from repayment_history;

-- ── IMPORT ORDER: borrowers → loan_accounts → repayment_history ──
SELECT COUNT(*) FROM borrowers;
SELECT COUNT(*) FROM loan_accounts;
SELECT COUNT(*) FROM repayment_history;
-- ── After Import verify ──── 
-- ════════════════════════════════════════════════════════════
--  QUERY 1: High-Risk Borrower Identification
--  Low credit score + high DTI + previous default
-- ════════════════════════════════════════════════════════════
SELECT
    borrower_id,
    credit_score,
    dti_ratio,
    previous_default,
    monthly_income,
    loan_amount,
    employment_type
FROM borrowers
WHERE credit_score < 600
  AND dti_ratio > 0.5
ORDER BY credit_score ASC, dti_ratio DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 2: Default Rate by Employment Type
-- ════════════════════════════════════════════════════════════
SELECT
    employment_type,
    COUNT(*)                                          AS total_borrowers,
    SUM(is_default)                                    AS defaults,
    ROUND(SUM(is_default) * 100.0 / COUNT(*), 2)       AS default_rate_pct,
    ROUND(AVG(credit_score), 0)                        AS avg_credit_score,
    ROUND(AVG(loan_amount), 2)                          AS avg_loan_amount
FROM borrowers
GROUP BY employment_type
ORDER BY default_rate_pct DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 3: Credit Score Band Risk Segmentation
--  Window function to bucket borrowers and rank risk
-- ════════════════════════════════════════════════════════════
WITH score_bands AS (
    SELECT *,
        CASE
            WHEN credit_score < 580 THEN 'Very Poor'
            WHEN credit_score < 670 THEN 'Fair'
            WHEN credit_score < 740 THEN 'Good'
            WHEN credit_score < 800 THEN 'Very Good'
            ELSE 'Excellent'
        END AS credit_band
    FROM borrowers
)
SELECT
    credit_band,
    COUNT(*)                                          AS total_borrowers,
    SUM(is_default)                                    AS defaults,
    ROUND(SUM(is_default) * 100.0 / COUNT(*), 2)       AS default_rate_pct,
    ROUND(AVG(loan_amount), 2)                          AS avg_loan_amount,
    RANK() OVER (ORDER BY SUM(is_default) * 1.0 / COUNT(*) DESC) AS risk_rank
FROM score_bands
GROUP BY credit_band
ORDER BY risk_rank;


-- ════════════════════════════════════════════════════════════
--  QUERY 4: Late Payment Pattern Detection
--  Borrowers with multiple late EMI payments
-- ════════════════════════════════════════════════════════════
SELECT
    r.borrower_id,
    b.credit_score,
    b.employment_type,
    COUNT(*)                                          AS total_emis,
    SUM(CASE WHEN r.payment_status = 'Late' THEN 1 ELSE 0 END) AS late_payments,
    ROUND(AVG(r.days_late), 1)                          AS avg_days_late
FROM repayment_history r
JOIN borrowers b ON r.borrower_id = b.borrower_id
GROUP BY r.borrower_id, b.credit_score, b.employment_type
HAVING late_payments >= 2
ORDER BY late_payments DESC, avg_days_late DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 5: Branch-wise NPA (Non-Performing Asset) Report
-- ════════════════════════════════════════════════════════════
SELECT
    la.branch,
    COUNT(*)                                          AS total_accounts,
    SUM(CASE WHEN la.account_status='NPA' THEN 1 ELSE 0 END) AS npa_accounts,
    ROUND(SUM(CASE WHEN la.account_status='NPA' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS npa_rate_pct,
    ROUND(SUM(la.disbursed_amount), 2)                  AS total_disbursed
FROM loan_accounts la
GROUP BY la.branch
ORDER BY npa_rate_pct DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 6: Loan Purpose Risk Analysis
-- ════════════════════════════════════════════════════════════
SELECT
    loan_purpose,
    COUNT(*)                                          AS total_loans,
    ROUND(AVG(loan_amount), 2)                          AS avg_loan_amount,
    SUM(is_default)                                    AS defaults,
    ROUND(SUM(is_default) * 100.0 / COUNT(*), 2)       AS default_rate_pct
FROM borrowers
GROUP BY loan_purpose
ORDER BY default_rate_pct DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 7: DTI Ratio vs Default Correlation
--  Bucketed analysis using CASE + GROUP BY
-- ════════════════════════════════════════════════════════════
SELECT
    CASE
        WHEN dti_ratio < 0.2 THEN '0-20%'
        WHEN dti_ratio < 0.35 THEN '20-35%'
        WHEN dti_ratio < 0.5 THEN '35-50%'
        ELSE '50%+'
    END AS dti_band,
    COUNT(*)                                          AS total_borrowers,
    SUM(is_default)                                    AS defaults,
    ROUND(SUM(is_default) * 100.0 / COUNT(*), 2)       AS default_rate_pct
FROM borrowers
GROUP BY dti_band
ORDER BY default_rate_pct DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 8: Monthly Loan Application & Default Trend
-- ════════════════════════════════════════════════════════════
SELECT
    DATE_FORMAT(application_date, '%Y-%m')              AS month,
    COUNT(*)                                            AS total_applications,
    SUM(is_default)                                      AS defaults,
    ROUND(SUM(is_default) * 100.0 / COUNT(*), 2)         AS default_rate_pct,
    ROUND(SUM(loan_amount), 2)                            AS total_loan_volume
FROM borrowers
GROUP BY DATE_FORMAT(application_date, '%Y-%m')
ORDER BY month;