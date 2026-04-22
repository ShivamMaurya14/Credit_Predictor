# 🚀 Production Deployment & Maintenance Guide

This guide details the steps required to deploy, maintain, and recalibrate the **Credit Risk Intelligence** pipeline. Following this guide ensures that the 125-feature parity and financial optimization remains intact across environments.

---

## 📋 System Requirements

### 💻 Hardware Specifications
- **Minimum**: 4GB RAM, Dual-core CPU.
- **Recommended**: 8GB+ RAM (to handle 307k record dataframes in the research notebook).
- **Disk**: 2GB of free space for datasets and model binaries.

### 🐍 Software Dependencies
- **Python**: 3.10 to 3.13.
- **Key Libraries**:
    - `FastAPI`, `Uvicorn`: For the production API.
    - `Scikit-learn`, `Imbalanced-learn`: For the ML engine.
    - `SHAP`: For local explainability.
    - `Joblib`: For model persistence.

---

## 🔧 Phase 1: Installation & Setup

1. **Clone & Initialize**:
   ```bash
   git clone <repository_url>
   cd Credit_Predictor
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Core Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Data Presence**:
   Ensure `datasets/application_train.csv` and `datasets/test.csv` are present. These are required for both training and bulk-lookup features.

---

## 🧪 Phase 2: Calibration & Synchronization (Mandatory)

The system relies on **Feature Parity**. If you retrain the model, you must update the production artifacts.

1. **Research Pipeline**:
   Open `nb/notebook.ipynb` and run the entire notebook.
   - **Step 1**: The notebook will calculate `GLOBAL_LOSS_PER_DEFAULT` dynamically.
   - **Step 2**: It will search for the optimal threshold (targeting the Global Max Benefit).
   - **Step 3**: It will export the 125-feature schema to `models/feature_names.json`.

2. **Artifact Export**:
   Ensure the following files are updated in the `models/` directory:
   - `random_forest_model.joblib` (The 31MB production binary).
   - `calibration_params.json` (Contains the Median/IQR for all 125 features used in `RobustScaler`).

---

## 🌐 Phase 3: Launching Production Services

### 1. Starting the API Server
Use `uvicorn` to launch the FastAPI application. We recommend port 8005 for internal service isolation.

```bash
uvicorn app:app --host 0.0.0.0 --port 8005 --workers 4
```

### 2. Validating the Startup
Watch the console logs. You should see:
- `✅ Random Forest model loaded from models/random_forest_model.joblib`
- `✅ Loaded 356,254 applicants with fully calculated 125-feature profiles`

---

## 🛠 Phase 4: Monitoring & Maintenance

### 🔍 API Endpoints Reference

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the Premium Dashboard. |
| `/predict` | `POST` | Processes a single applicant JSON with 125 features. |
| `/upload_csv` | `POST` | Batch processes large CSV files for portfolio analysis. |
| `/applicant/{id}` | `GET` | Retrieves pre-calculated data for an existing applicant. |
| `/docs` | `GET` | Interactive Swagger/OpenAPI documentation. |

### 📈 Model Drifting & Recalibration
Financial markets shift. We recommend a **Quarterly Recalibration**:
1. Update the `datasets/` folder with fresh loan data.
2. Run the `nb/notebook.ipynb` to find the new optimal threshold.
3. Replace the `calibration_params.json` to ensure scaling parity.

### 🛡 Audit Logs
All decisions are logged to `history.db`. This SQLite database stores:
- Timestamp of assessment.
- Applicant ID.
- Predicted Probability.
- Final Decision (Approved/Rejected) based on the threshold.

---

## 🚦 Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **500 Error on Predict** | Feature Mismatch | Ensure `feature_names.json` matches the input data. |
| **TypeError in SHAP** | Format Change | Updated `app.py` handles 3D Random Forest outputs. |
| **Port 8005 Occupied** | Server Zombie | Use `lsof -i :8005` to find and kill the process. |
| **Low Benefit Result** | Threshold Mismatch | Verify `OPTIMAL_THRESHOLD` in `app.py` matches the notebook result. |

---

**Guide Version**: 1.0 
**Maintenance Contact**: Shivam Maurya
