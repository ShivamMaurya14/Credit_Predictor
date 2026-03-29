# 🚀 Model Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Random Forest credit scoring model in production environments.

## ✅ Prerequisites

- Python 3.8+
- Required packages: See `requirements.txt`
- 31 MB disk space for Random Forest model
- Input data in correct format with all 124 features

## 📦 Model Selection

### Recommended Model: Random Forest
- **Why**: Highest ROC-AUC (0.8866) and Net Benefit ($9.1B)
- **File**: `models/random_forest_model.joblib`
- **Optimal Threshold**: 0.40
- **Expected Performance**: Catches 91.5% of defaults

### Alternative Models
- **LightGBM** (`lightgbm_model.txt`): Lower false positives, higher operational cost
- **XGBoost** (`xgboost_model.json`): Balanced performance

---

## 🔧 Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Input features: **124 features** after preprocessing

Required preprocessing:
```python
from sklearn.preprocessing import RobustScaler
import joblib

# Load the pre-trained scaler (same one used in training)
scaler = joblib.load('models/feature_scaler.joblib')

# Apply scaling to your input data
X_scaled = scaler.transform(X)
```

### 3. Model Loading

```python
import joblib

# Load the Random Forest model
model = joblib.load('models/random_forest_model.joblib')

# Load feature names for validation
import json
with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)
```

---

## 🎯 Making Predictions

### Batch Prediction

```python
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model = joblib.load('models/random_forest_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')

# Assuming you have a dataframe 'df' with 124 features
X_scaled = scaler.transform(df)

# Get probability predictions
probabilities = model.predict_proba(X_scaled)[:, 1]

# Apply optimal threshold (0.40)
threshold = 0.40
predictions = (probabilities > threshold).astype(int)

# Add to results
df['default_probability'] = probabilities
df['default_prediction'] = predictions
```

### Single Prediction

```python
# Single record as numpy array (shape: 1, 124)
X_single = scaler.transform([single_record])

# Get default probability
prob = model.predict_proba(X_single)[0, 1]

# Decision
is_default_risk = prob > 0.40

print(f"Default Probability: {prob:.4f}")
print(f"Default Risk: {is_default_risk}")
```

---

## 📊 Performance Expectations

### Model Performance Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8866 |
| Recall | 91.46% |
| Precision | 20.85% |
| F1-Score | 0.3395 |
| Defaults Caught | 22,704 / 24,825 (91.5%) |

### Financial Impact

- **Annual Benefit**: $9,085.84 Million
- **Loss Prevented**: $11,659.21 Million
- **False Positives**: 86,207 (manual review needed)
- **Opportunity Cost**: $2,573.37 Million

---

## 🔍 Model Interpretation

### Prediction Output

The model returns probabilities between 0 and 1:
- **0.0 - 0.40**: Low default risk (approve)
- **0.40 - 1.00**: High default risk (further review)

### Decision Making

**At Optimal Threshold (0.40):**
- Captures 91.5% of actual defaults
- Rejects 30.5% of all applications (22,704 + 86,207 / 307,511)
- Prevents ~$11.6B in default losses

---

## ⚠️ Important Considerations

### Data Quality
- Ensure all 124 features are present
- Handle missing values (median imputation recommended)
- Use the exact same preprocessing steps as training
- Verify feature order matches training data

### Model Limitations
- Trained on historical Home Credit data
- Performance may vary with different populations
- Requires regular retraining (recommend quarterly)
- Threshold may need adjustment based on business needs

### Monitoring

Track these metrics in production:
1. **Distribution shift**: Compare input feature distributions vs training data
2. **Model performance**: Monitor actual default rates vs predictions
3. **Threshold effectiveness**: Track defaults caught vs false positives
4. **Business metrics**: Monitor revenue impact

---

## 🔄 Retraining Pipeline

### When to Retrain
- After 6 months of production use
- When model performance drops > 5%
- When significant population shift detected
- After major business/policy changes

### Retraining Steps
1. Collect new labeled data
2. Run complete preprocessing pipeline
3. Train all 3 models (RF, LightGBM, XGBoost)
4. Compare performance metrics
5. A/B test winner on production data
6. Deploy with versioning

---

## 🐛 Troubleshooting

### Issue: Model Loading Error
```python
# Verify model file exists and is valid
import os
print(os.path.exists('models/random_forest_model.joblib'))

# Try reloading
import joblib
try:
    model = joblib.load('models/random_forest_model.joblib')
except Exception as e:
    print(f"Error: {e}")
```

### Issue: Dimension Mismatch
```python
# Verify feature count
print(f"Features in data: {X.shape[1]}")  # Should be 124
print(f"Features expected: {model.n_features_in_}")
```

### Issue: Scaling Problems
```python
# Ensure using the correct scaler
scaler = joblib.load('models/feature_scaler.joblib')
print(f"Scaler type: {type(scaler)}")

# Check scaling statistics
print(f"Scaler center: {scaler.center_[:5]}")
print(f"Scaler scale: {scaler.scale_[:5]}")
```

---

## 📋 Deployment Checklist

- [ ] Requirements installed: `pip install -r requirements.txt`
- [ ] Model file exists: `models/random_forest_model.joblib`
- [ ] Scaler file exists: `models/feature_scaler.joblib`
- [ ] Feature names loaded: `models/feature_names.json`
- [ ] Input data preprocessing validated
- [ ] Model performance tested on sample data
- [ ] Threshold set to 0.40
- [ ] Logging configured
- [ ] Monitoring dashboard set up
- [ ] Backup model identified (LightGBM)

---

## 🚀 Deployment Options

### Option 1: REST API (Flask)

```python
from flask import Flask, request, jsonify
import joblib
import json

app = Flask(__name__)

# Load models once at startup
model = joblib.load('models/random_forest_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    X_scaled = scaler.transform([data])
    probability = model.predict_proba(X_scaled)[0, 1]
    
    return jsonify({
        'probability': float(probability),
        'default_risk': probability > 0.40
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Option 2: Batch Processing

```python
import pandas as pd
import joblib

# Load data
df = pd.read_csv('input_data.csv')

# Load model & scaler
model = joblib.load('models/random_forest_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')

# Scale and predict
X_scaled = scaler.transform(df)
probabilities = model.predict_proba(X_scaled)[:, 1]

# Save results
results = pd.DataFrame({
    'customer_id': df.index,
    'default_probability': probabilities,
    'default_risk': probabilities > 0.40
})

results.to_csv('predictions.csv', index=False)
```

### Option 3: Docker Container

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY app.py .

EXPOSE 5000
CMD ["python", "app.py"]
```

---

## 📞 Support & Issues

For deployment issues:
1. Check model file integrity
2. Verify feature count (should be 124)
3. Review preprocessing steps
4. Consult `DEPLOYMENT_MANIFEST.json` for metadata
5. Review original notebook for context

---

**Last Updated**: March 29, 2026
**Model Version**: CREDIT_SCORING_MODELS_v1
**Status**: Production Ready ✅
