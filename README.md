# 💳 Credit Risk Intelligence System (Detection Rate **91.5%**)

**Alternate Credit Scoring System for Unbanked Users Using Dynamic Threshold Optimization.**

An advanced machine learning solution that predicts credit default risk for unbanked and thin-file users using the Home Credit dataset (307,511 records). Optimized for **Portfolio Profitability**, this system maximizes the "Net Financial Benefit" through synchronized feature parity between research and production.

---

## 🎯 Problem Statement

- **Objective**: Predict credit default risk for unbanked users
- **Dataset**: 307,511 loan applications from Home Credit
- **Challenge**: Severe class imbalance (only 8% defaults vs 92% repaid)
- **Goal**: Build cost-optimized models that minimize financial losses while maximizing default detection

### Data Overview
- **Total Records**: 307,511 (Training)
- **Class Distribution**:
  - Non-defaulters: 282,686 (91.9%)
  - Defaulters: 24,825 (8.1%)
  - Imbalance Ratio: **11.39:1**

---

## 🔍 Solution Approach: Profit-First ML

This project implements a **Synchronized Pipeline** where the research environment dynamically calibrates the production environment.

### Model Evolution & Strategy
1. **🏆 Random Forest (Winner)**: 200 trees with `class_weight='balanced'`. Optimized for the highest Net Financial Benefit.
2. **LightGBM + SMOTE**: Gradient boosting with synthetic oversampling for aggressive risk capture.
3. **XGBoost (Cost-Sensitive)**: Advanced ensemble using `scale_pos_weight` for loss minimization.

### Key Techniques Applied
- ✅ **125-Feature Engineering**: 13 unique credit-specific metrics, including `DAYS_EMPLOYED_PERCENT`.
- ✅ **Dynamic Calibration**: Automated calculation of `Loss per Default` ($513k) and `Opportunity Cost` ($30k).
- ✅ **Robust Preprocessing**: Manual `RobustScaler` implementation using synchronized `calibration_params.json`.
- ✅ **Threshold Optimization**: Searching the Global Maximum for Portfolio Utility (Winner at 0.40).
- ✅ **SHAP Explainability**: 1D feature impact analysis for regulatory reason-codes.

---

## 📊 FEATURE ENGINEERING & SCALING

### Preprocessing Steps

1. **Outlier Detection**: IQR clipping with 3.0× tolerance on 8 numeric columns
2. **Missing Values**: Median imputation for numeric features
3. **Categorical Encoding**: LabelEncoder + one-hot encoding for categorical features
4. **Feature Engineering**: 13 credit-specific features created

#### Critical Domain Features Created
```text
• DAYS_EMPLOYED_PERCENT - Employment duration relative to age (125th Feature)
• CREDIT_INCOME_RATIO    - Total loan amount relative to annual earnings
• ANNUITY_INCOME_RATIO   - Monthly payment sustainability
• EXT_SOURCE_AVG         - Aggregated agency scores
```

### Feature Parity Summary
| Phase | Features | Status |
| :--- | :--- | :--- |
| **Raw Data** | 122 | Baseline |
| **Post-Engineering** | 140 | Expanded |
| **Final Production** | 125 | Optimized (Deduplicated) |

---

## 🤖 MODEL PERFORMANCE COMPARISON

### Financial Optimization Logic
Models were optimized to minimize the **Total Portfolio Loss**:
- **Loss Per Default**: Median `AMT_CREDIT` of defaulters (~$513k).
- **Opportunity Cost**: 10% of the median annual annuity for good clients (~$30k).

### 🥇 FINAL RESULTS (Ranked by Net Benefit)

#### 1️⃣ Random Forest (PRODUCTION WINNER) ⭐⭐⭐
- **ROC-AUC**: 0.8866
- **Recall (Detection)**: **91.46%**
- **Optimal Threshold**: **0.40**
- **Net Financial Benefit**: **~$9.08 Billion** 💰

#### 2️⃣ LightGBM + SMOTE ⭐⭐
- **ROC-AUC**: 0.7813
- **Detection Rate**: 89.73%
- **Optimal Threshold**: 0.05
- **Net Benefit**: ~$6.77 Billion

#### 3️⃣ XGBoost (Cost-Sensitive) ⭐
- **ROC-AUC**: 0.7647
- **Detection Rate**: 81.9%
- **Optimal Threshold**: 0.23
- **Net Benefit**: ~$6.63 Billion

---

## 📂 Project Structure

```text
.
├── app.py                # Asynchronous FastAPI Inference Engine
├── nb/
│   └── notebook.ipynb    # Main Analysis, Calibration & Research
├── datasets/
│   └── test.csv          # Evaluation & Bulk Lookup Dataset
├── models/
│   ├── random_forest_model.joblib  # Production Model Binary
│   ├── feature_names.json          # 125-feature schema
│   └── calibration_params.json     # RobustScaler Statistics (Sync'd)
├── static/               # Premium Dark-Mode Dashboard
│   ├── index.html        # UI Skeleton (Glassmorphism)
│   ├── script.js         # Frontend Logic (SHAP Integration)
│   └── style.css         # UI Aesthetics
├── DEPLOYMENT_GUIDE.md    # Installation & Calibration Instructions
├── ARCHITECTURE.md        # Technical Design & Data Flow
├── requirements.txt       # Python Dependencies
└── history.db             # SQLite Production Audit Log
```

---

## 🚀 QUICK START GUIDE

### 1. Installation
```bash
# Clone and enter directory
pip install -r requirements.txt
```

### 2. Calibration (Ensuring Parity)
Open `nb/notebook.ipynb` and **Run All**. This will:
- Calibrate the `RobustScaler` on the training set.
- Find the most profitable threshold.
- Export the latest `calibration_params.json`.

### 3. Launch the Dashboard
```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8005
```
Access the dashboard at `http://localhost:8005`.

---

## 💡 Key Technical Pillars

1. **Feature Synchronization**: The 125-feature engineering in `app.py` is an exact manual replica of the `notebook.ipynb` logic to ensure zero prediction drift.
2. **Outlier Resistance**: By using `RobustScaler` (Median/IQR), the model remains stable even when applicants report extremely high or low incomes.
3. **AI Explainability**: Every prediction returns the top 6 SHAP factors, explaining why a loan was approved or rejected (e.g., "High income decreases risk", "Low agency scores increase risk").

---

**Last Updated**: April 22, 2026   
**Author**: Shivam Maurya  
