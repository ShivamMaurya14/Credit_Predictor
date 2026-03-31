# Model Architecture & Technical Design

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  INPUT DATA (307,511 records)           │
│                    121 raw features                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              DATA PREPROCESSING LAYER                   │
│  • Categorical encoding (LabelEncoder + OneHot)         │
│  • Outlier detection (IQR clipping, 3.0x tolerance)    │
│  • Missing value imputation (Median)                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│        FEATURE ENGINEERING LAYER (124 features)        │
│  • Credit-to-Income ratio                              │
│  • Annuity-to-Income ratio                            │
│  • Credit term metrics                                 │
│  • Per capita income                                   │
│  • 10+ additional domain features                      │
│  • Removed 15 highly correlated features               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           FEATURE SCALING LAYER                         │
│        (RobustScaler - resistant to outliers)          │
│        • Applied to all 124 features                   │
│        • Saved in: feature_scaler.joblib              │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌────────┐   ┌──────────┐   ┌────────┐
    │ Random │   │ LightGBM │   │XGBoost │
    │ Forest │   │+ SMOTE   │   │        │
    └────────┘   └──────────┘   └────────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  THRESHOLD OPTIMIZATION │
         │  (Loss minimization)    │
         │  RF: 0.40               │
         │  LGBM: 0.05             │
         │  XGB: 0.23              │
         └────────────┬────────────┘
                      │
                      ▼
          ┌──────────────────────┐
          │ FINAL PREDICTIONS &  │
          │ BUSINESS METRICS     │
          └──────────────────────┘
```

---

## Model Specifications

### Random Forest (WINNER)

**Architecture:**
```
RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=15,            # Depth limit to prevent overfitting
    min_samples_split=50,    # Minimum samples required to split
    min_samples_leaf=20,     # Minimum samples required at leaf
    class_weight='balanced', # Handle class imbalance
    n_jobs=-1,              # Use all CPU cores
    random_state=42         # Reproducibility
)
```

**Training:**
- Training samples: 307,511
- Features: 124
- Classes: 2 (default/non-default)
- Training time: ~5-10 minutes on standard hardware
- ROC-AUC achieved: 0.8866

---

### LightGBM with SMOTE

**Architecture:**
```
LGBMClassifier(
    n_estimators=200,       # 200 boosting rounds
    scale_pos_weight=6.19,  # Cost-sensitive weights
    is_unbalanced=True,
    SMOTE applied to training data
)
```

**Training Process:**
1. Original data: 307,511 samples (24,825 positive)
2. After SMOTE: 565,372 samples (balanced)
3. Gradient boosting optimization
4. Training time: ~15-20 minutes

**Formats:**
- Primary: `lightgbm_model.txt` (native format)
- Backup: `lightgbm_model_backup.joblib`

---

### XGBoost with Cost-Sensitive Learning

**Architecture:**
```
XGBClassifier(
    n_estimators=300,              # 300 boosting rounds
    scale_pos_weight=6.19,         # Cost weighting for class imbalance
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)
```

**Training:**
- 5-Fold Stratified Cross-Validation
- Mean CV ROC-AUC: 0.7647 ± 0.0043
- Training time: ~20-30 minutes with 5-fold CV

**Formats:**
- Primary: `xgboost_model.json` (cross-platform)
- Production: `xgboost_model.ubj` (binary, faster loading)

---

## Key Techniques & Improvements

### 1. Class Imbalance Handling

**Problem:** Only 8.1% of applications are defaults (11.39:1 imbalance)

**Solutions Applied:**
- **Random Forest**: `class_weight='balanced'`
  - Automatically weights classes inversely proportional to frequency
  - Effect: Increased default detection from ~60% to ~91%

- **LightGBM**: SMOTE oversampling
  - Creates synthetic minority samples
  - Balances dataset to 50-50 before training
  - Slight increase in false positives but better default capture

- **XGBoost**: Cost-sensitive learning
  - `scale_pos_weight=6.19` (11.39 ÷ 2 for better performance)
  - Effectively teaches model to care more about defaults

### 2. Threshold Optimization

**Standard Approach:** Use 0.50 probability threshold
- Catches: ~60% of defaults (too many missed)
- False positives: Too high still

**Our Approach:** Find threshold that minimizes financial loss
```
Net Benefit = (Defaults Caught × Loss Per Default) 
            - (False Positives × Cost Per Rejected Good)
```

**Results:**
- Random Forest: Optimal threshold = 0.40 (better recall)
- LightGBM: Optimal threshold = 0.05 (very aggressive)
- XGBoost: Optimal threshold = 0.23 (balanced)

### 3. Feature Engineering

**Created credit-specific features:**
```
CREDIT_INCOME_RATIO = AMT_CREDIT / (AMT_INCOME_TOTAL + 1)
    ↳ Measures loan burden relative to income

ANNUITY_INCOME_RATIO = AMT_ANNUITY / (AMT_INCOME_TOTAL + 1)
    ↳ Measures monthly payment relative to income

CREDIT_TERM = AMT_ANNUITY / (AMT_CREDIT + 1)
    ↳ Implicit interest rate / duration

INCOME_PER_PERSON = AMT_INCOME_TOTAL / max(CNT_FAM_MEMBERS, 1)
    ↳ Per capita household income
```

**Feature Selection:**
- Initial features: 140 (after encoding + engineering)
- Removed: 15 highly correlated features (r > 0.95)
- Final features: 124 (optimal balance)

### 4. Robust Scaling

**Why RobustScaler?**
- Resistant to outliers (uses median & IQR)
- Suitable for financial data with extreme values
- Preserves information better than StandardScaler

**Scaling formula:**
```
X_scaled = (X - median) / IQR
```

---

## Model Evaluation Metrics

### Standard ML Metrics

| Metric | Formula | What it measures |
|--------|---------|------------------|
| ROC-AUC | Area under ROC curve | Overall discrimination ability |
| Recall | TP/(TP+FN) | % of defaults caught |
| Precision | TP/(TP+FP) | % of predictions correct |
| F1-Score | 2(P·R)/(P+R) | Harmonic mean of P & R |
| Accuracy | (TP+TN)/Total | Overall correctness |

### Business Metrics (Custom)

```python
def calculate_net_benefit(y_true, y_pred_prob, threshold):
    """
    Calculate financial impact of predictions
    """
    predictions = y_pred_prob > threshold
    
    # Confusion matrix
    TP = np.sum((predictions == 1) & (y_true == 1))
    FP = np.sum((predictions == 1) & (y_true == 0))
    
    # Financial calculations
    loss_prevented = TP * LOSS_PER_DEFAULT  # $513,531
    opportunity_cost = FP * COST_PER_REJECTED_GOOD  # $29,851
    
    net_benefit = loss_prevented - opportunity_cost
    
    return {
        'defaults_caught': TP,
        'false_positives': FP,
        'loss_prevented': loss_prevented,
        'opportunity_cost': opportunity_cost,
        'net_benefit': net_benefit
    }
```

---

## Performance Analysis

### Distribution of Predictions

**Random Forest - Probability Distribution:**
- Non-defaulters: Median prob = 0.15, Q95 = 0.35
- Defaulters: Median prob = 0.55, Q95 = 0.85
- Good class separation allows low false positive rate

**At threshold 0.40:**
- Catches 91.5% of true defaults
- Rejects 30.5% of all applicants (including 86k good ones)
- Net benefit: $9.1B annually

---

## Model Comparison - Why Random Forest Won

| Factor | Random Forest | LightGBM | XGBoost |
|--------|-------------|----------|---------|
| ROC-AUC | **0.8866** | 0.7813 | 0.7647 |
| Defaults Caught (%) | **91.5** | 89.7 | 81.9 |
| False Positives | **86,207** | 156,468 | 127,598 |
| Net Benefit ($M) | **9,085.84** | 6,768.18 | 6,627.56 |
| Training Time | Fast | Medium | Slow (CV) |
| Interpretability | High | Medium | Low |
| Optimal Threshold | 0.40 | 0.05 | 0.23 |

**Winner's Advantages:**
1. Highest discrimination (ROC-AUC 0.8866)
2. Best financial outcome ($9.1B)
3. Practical threshold (0.40)
4. Fast inference time
5. Robust to feature scaling
6. Built-in feature importance

---

## Data Flow Diagram

```
Training Pipeline:
┌──────────────────┐
│  Raw Data        │
│  307,511 × 122   │ 
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│ Preprocessing            │
│ • Encoding               │
│ • Imputation             │
│ • Outlier handling       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Feature Engineering      │
│ • Create 4 domain feats  │
│ • Remove correlated (15) │
│ • Final: 124 features    │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Data Scaling             │
│ • RobustScaler applied   │
│ • 307,511 × 124 matrix   │
└────────┬─────────────────┘
         │
    ┌────┴────────┬────────┐
    │             │        │
    ▼             ▼        ▼
 RF Model    LGBM Model  XGB Model
    │             │        │
    └────────┬────┴────┬───┘
             │        │
             ▼        ▼
        Threshold Optimization
             │        │
             └────┬───┘
                  ▼
        Deployment & Evaluation
```

---

## Production Considerations

### Latency Requirements
- Batch prediction: < 5 minutes for 100k samples
- Real-time prediction: < 100ms per sample
- API response time: < 500ms

### Scalability
- Model size: 31 MB (fits in memory)
- Inference parallelizable
- Can handle 10k+ predictions/second

### Monitoring
- Track prediction distribution over time
- Monitor actual vs predicted defaults
- Alert on significant distribution shift
- Log all predictions for audit trail

---

## Future Improvements

1. **Ensemble stacking** - Combine all 3 models for better performance
2. **Hyperparameter tuning** - GridSearch/BayesianOptimization
3. **Deep learning** - Neural networks for complex patterns
4. **Time-series features** - Incorporate temporal patterns
5. **Feature interactions** - Polynomial features, interaction terms
6. **Online learning** - Continuous model updates
7. **Explainability** - SHAP/LIME for local interpretability

---

**Architecture Last Updated**: March 29, 2026
**Complexity**: Production-Ready
**Scalability**: For 300k+ daily predictions
