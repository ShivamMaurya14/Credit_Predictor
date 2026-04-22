from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import shap
import os
import io
import sqlite3
from datetime import datetime

app = FastAPI(title="Credit Default Predictor API")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load feature names
FEATURE_NAMES_PATH = "models/feature_names.json"
MODEL_PATH = "models/random_forest_model.joblib"
OPTIMAL_THRESHOLD = 0.40  # Best threshold for Random Forest as per notebook calibration

feature_names = []
if os.path.exists(FEATURE_NAMES_PATH):
    with open(FEATURE_NAMES_PATH, "r") as f:
        data = json.load(f)
        feature_names = data["feature_names"] if isinstance(data, dict) and "feature_names" in data else data
else:
    print(f"Warning: {FEATURE_NAMES_PATH} not found.")

# Database Setup
DB_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  app_id TEXT,
                  probability REAL,
                  is_default INTEGER,
                  income REAL,
                  credit REAL)''')
    conn.commit()
    conn.close()

init_db()

def save_prediction(app_id, prob, is_default, income, credit):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO predictions (timestamp, app_id, probability, is_default, income, credit) VALUES (?, ?, ?, ?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), app_id, prob, 1 if is_default else 0, income, credit))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to DB: {e}")

# Global model and explainer
model = None
explainer = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        # Random Forest SHAP Explainer (TreeExplainer)
        explainer = shap.TreeExplainer(model)
        print(f"✅ Random Forest model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file {MODEL_PATH} not found.")

# Global applicants database
applicants_db = {}

# Calibration params
CALIBRATION_PATH = "models/calibration_params.json"
calibration_params = {}
if os.path.exists(CALIBRATION_PATH):
    with open(CALIBRATION_PATH, 'r') as f:
        calibration_params = json.load(f)

def load_csv_to_db(df):
    global applicants_db
    if "SK_ID_CURR" not in df.columns:
        print("ERROR: SK_ID_CURR not in CSV")
        return
    
    # Pre-process for numeric encoding
    working_df = df.copy()
    
    # Basic data cleaning
    working_df = working_df.dropna(subset=["SK_ID_CURR"])
    working_df['DAYS_EMPLOYED'] = working_df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # CALCULATE ENGINEERED FEATURES (Matching training logic)
    working_df['CREDIT_INCOME_RATIO'] = working_df['AMT_CREDIT'] / (working_df['AMT_INCOME_TOTAL'] + 1)
    working_df['ANNUITY_INCOME_RATIO'] = working_df['AMT_ANNUITY'] / (working_df['AMT_INCOME_TOTAL'] + 1)
    working_df['CREDIT_TERM'] = working_df['AMT_ANNUITY'] / (working_df['AMT_CREDIT'] + 1)
    working_df['INCOME_PER_PERSON'] = working_df['AMT_INCOME_TOTAL'] / (working_df['CNT_FAM_MEMBERS'].fillna(2).replace(0, 1))
    working_df['LOAN_BURDEN_RATIO'] = working_df['AMT_ANNUITY'] / (working_df['AMT_INCOME_TOTAL'] + 1)
    working_df['LOAN_TO_VALUE'] = working_df['AMT_CREDIT'] / (working_df['AMT_GOODS_PRICE'].fillna(working_df['AMT_CREDIT']) + 1)
    working_df['EMPLOYMENT_STABILITY'] = working_df['DAYS_EMPLOYED'].abs() / 365
    working_df['YEARS_EMPLOYED'] = working_df['DAYS_EMPLOYED'].abs() / 365
    working_df['AGE_YEARS'] = working_df['DAYS_BIRTH'].abs() / 365
    
    # Asset Mappings
    asset_map = {'Y': 1, 'N': 0}
    working_df['FLAG_OWN_CAR'] = working_df['FLAG_OWN_CAR'].map(asset_map).fillna(0)
    working_df['FLAG_OWN_REALTY'] = working_df['FLAG_OWN_REALTY'].map(asset_map).fillna(0)
    
    # Age Risk Segment
    working_df['AGE_RISK_SEGMENT'] = pd.cut(working_df['AGE_YEARS'], bins=[0, 25, 35, 50, 65, 120], labels=[5, 4, 3, 2, 1]).astype(float)
    
    working_df['CREDIT_PER_FAMILY'] = working_df['AMT_CREDIT'] / (working_df['CNT_FAM_MEMBERS'].fillna(1) + 1)
    working_df['INCOME_PER_FAMILY'] = working_df['AMT_INCOME_TOTAL'] / (working_df['CNT_FAM_MEMBERS'].fillna(1) + 1)
    working_df['IMPLIED_LOAN_TERM_YEARS'] = working_df['AMT_CREDIT'] / (working_df['AMT_ANNUITY'] + 1)

    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    working_df['EXT_SOURCE_AVG'] = working_df[ext_cols].mean(axis=1)
    working_df['EXT_SOURCE_MEDIAN'] = working_df[ext_cols].median(axis=1)
    working_df['EXT_SOURCE_STD'] = working_df[ext_cols].std(axis=1)
    
    working_df['DEBT_TO_INCOME'] = (working_df['AMT_ANNUITY'] * 12) / (working_df['AMT_INCOME_TOTAL'] + 1)
    working_df['INCOME_RELIABILITY'] = 1.0 / (1.0 + working_df['DAYS_LAST_PHONE_CHANGE'].abs() / 365)
    working_df['TOTAL_MONTHLY_OBLIGATION'] = working_df['AMT_ANNUITY'] * 1.1
    working_df['DAYS_EMPLOYED_PERCENT'] = working_df['DAYS_EMPLOYED'] / (working_df['DAYS_BIRTH'] + 1)

    # Handle string columns for the model
    # Mimic LabelEncoder by sorting unique values alphabetically
    for col in working_df.select_dtypes('object').columns:
        unique_vals = sorted(working_df[col].astype(str).unique())
        val_map = {val: i for i, val in enumerate(unique_vals)}
        working_df[col] = working_df[col].astype(str).map(val_map)

    # Deduplicate feature names
    model_features = list(dict.fromkeys(feature_names))
    # Ensure UI fields are also kept
    ui_fields = ["YEARS_EMPLOYED", "AGE_YEARS", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CODE_GENDER", "NAME_EDUCATION_TYPE", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    
    available_features = [f for f in model_features if f in working_df.columns and f != "SK_ID_CURR"]
    for f in ui_fields:
        if f not in available_features and f in working_df.columns:
            available_features.append(f)
    
    # Store records
    records = working_df[["SK_ID_CURR"] + available_features].to_dict('records')
    for rec in records:
        app_id = str(int(rec["SK_ID_CURR"]))
        applicants_db[app_id] = rec
    
    print(f"Loaded {len(applicants_db)} applicants with fully calculated 125-feature profiles")

# Try to load existing dataset on startup
DEFAULT_CSV = "datasets/test.csv"
if os.path.exists(DEFAULT_CSV):
    try:
        df_init = pd.read_csv(DEFAULT_CSV)
        load_csv_to_db(df_init)
    except Exception as e:
        print(f"Error loading {DEFAULT_CSV}: {e}")

# Data models
class PredictionRequest(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    AGE_YEARS: float
    YEARS_EMPLOYED: float
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    NAME_EDUCATION_TYPE: int
    CNT_CHILDREN: int
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    CNT_FAM_MEMBERS: Optional[float] = 2.0
    DAYS_LAST_PHONE_CHANGE: Optional[float] = 0.0
    app_id: Optional[str] = "Manual"

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        load_csv_to_db(df)
        return {"message": "CSV processed"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

@app.get("/applicant/{app_id}")
async def get_applicant(app_id: str):
    if app_id not in applicants_db:
        # Try numeric matching if string match fails
        try:
            numeric_id = float(app_id)
            for k, v in applicants_db.items():
                if float(k) == numeric_id:
                    app_id = k
                    break
            else:
                raise HTTPException(status_code=404, detail="Applicant not found")
        except:
            raise HTTPException(status_code=404, detail="Applicant not found")
    
    data = applicants_db[app_id].copy()
    # Clean NaN for JSON compliance
    for k, v in data.items():
        if isinstance(v, float) and np.isnan(v):
            data[k] = None
    return data

@app.post("/predict")
async def predict(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 1. Initialize feature vector with NaN
    features_vector = np.full(len(feature_names), np.nan)
    
    # 2. Start with baseline data if app_id exists
    applicant_data = {}
    if req.app_id and req.app_id in applicants_db:
        applicant_data = applicants_db[req.app_id].copy()
    
    # 3. Update with form inputs (User's "What-If" changes)
    # Map form fields back to original column names
    form_mappings = {
        "AMT_INCOME_TOTAL": req.AMT_INCOME_TOTAL,
        "AMT_CREDIT": req.AMT_CREDIT,
        "AMT_ANNUITY": req.AMT_ANNUITY,
        "AMT_GOODS_PRICE": req.AMT_GOODS_PRICE,
        "DAYS_BIRTH": -(req.AGE_YEARS * 365.25),
        "DAYS_EMPLOYED": -(req.YEARS_EMPLOYED * 365.25),
        "CODE_GENDER": req.CODE_GENDER,
        "FLAG_OWN_CAR": req.FLAG_OWN_CAR,
        "FLAG_OWN_REALTY": req.FLAG_OWN_REALTY,
        "NAME_EDUCATION_TYPE": req.NAME_EDUCATION_TYPE,
        "CNT_CHILDREN": req.CNT_CHILDREN,
        "CNT_FAM_MEMBERS": req.CNT_FAM_MEMBERS,
        "DAYS_LAST_PHONE_CHANGE": req.DAYS_LAST_PHONE_CHANGE,
        "EXT_SOURCE_1": req.EXT_SOURCE_1 / 100.0,
        "EXT_SOURCE_2": req.EXT_SOURCE_2 / 100.0,
        "EXT_SOURCE_3": req.EXT_SOURCE_3 / 100.0,
    }
    
    # Apply mappings
    for col, val in form_mappings.items():
        applicant_data[col] = val

    # 4. Calculate Engineered Features
    s1, s2, s3 = applicant_data.get("EXT_SOURCE_1", 0.5), applicant_data.get("EXT_SOURCE_2", 0.5), applicant_data.get("EXT_SOURCE_3", 0.5)
    ext_vals = [s1, s2, s3]
    
    # Calculate domain features
    applicant_data["CREDIT_INCOME_RATIO"] = req.AMT_CREDIT / (req.AMT_INCOME_TOTAL + 1)
    applicant_data["ANNUITY_INCOME_RATIO"] = req.AMT_ANNUITY / (req.AMT_INCOME_TOTAL + 1)
    applicant_data["CREDIT_TERM"] = req.AMT_ANNUITY / (req.AMT_CREDIT + 1)
    applicant_data["DAYS_EMPLOYED_PERCENT"] = (-(req.YEARS_EMPLOYED * 365.25)) / ((-(req.AGE_YEARS * 365.25)) + 1)
    applicant_data["INCOME_PER_PERSON"] = req.AMT_INCOME_TOTAL / max(1, req.CNT_FAM_MEMBERS)
    applicant_data["LOAN_BURDEN_RATIO"] = req.AMT_ANNUITY / (req.AMT_INCOME_TOTAL + 1)
    applicant_data["LOAN_TO_VALUE"] = req.AMT_CREDIT / (req.AMT_GOODS_PRICE + 1)
    applicant_data["EMPLOYMENT_STABILITY"] = req.YEARS_EMPLOYED
    applicant_data["AGE_YEARS"] = req.AGE_YEARS
    
    age = req.AGE_YEARS
    applicant_data["AGE_RISK_SEGMENT"] = 5 if age <= 25 else (4 if age <= 35 else (3 if age <= 50 else (2 if age <= 65 else 1)))
    
    applicant_data["CREDIT_PER_FAMILY"] = req.AMT_CREDIT / (req.CNT_FAM_MEMBERS + 1)
    applicant_data["INCOME_PER_FAMILY"] = req.AMT_INCOME_TOTAL / (req.CNT_FAM_MEMBERS + 1)
    applicant_data["IMPLIED_LOAN_TERM_YEARS"] = req.AMT_CREDIT / (req.AMT_ANNUITY + 1)
    
    applicant_data["EXT_SOURCE_AVG"] = np.mean(ext_vals)
    applicant_data["EXT_SOURCE_MEDIAN"] = np.median(ext_vals)
    applicant_data["EXT_SOURCE_STD"] = np.std(ext_vals)
    
    applicant_data["DEBT_TO_INCOME"] = (req.AMT_ANNUITY * 12) / (req.AMT_INCOME_TOTAL + 1)
    applicant_data["INCOME_RELIABILITY"] = 1.0 / (1.0 + abs(req.DAYS_LAST_PHONE_CHANGE) / 365)
    applicant_data["TOTAL_MONTHLY_OBLIGATION"] = req.AMT_ANNUITY * 1.1

    # 5. Populate features vector and apply scaling (RobustScaler)
    for i, feat_name in enumerate(feature_names):
        val = applicant_data.get(feat_name, np.nan)
        if pd.isna(val):
            features_vector[i] = np.nan
        else:
            if feat_name in calibration_params and "median" in calibration_params[feat_name]:
                median = calibration_params[feat_name]["median"]
                iqr = calibration_params[feat_name]["iqr"]
                # Apply RobustScaler: (x - median) / iqr
                features_vector[i] = (val - median) / iqr
            else:
                features_vector[i] = val
    
    # 6. Reshape and predict
    # 6. Predict using Random Forest
    input_array = features_vector.reshape(1, -1)
    # Median imputation for any remaining NaNs (Random Forest doesn't handle them automatically)
    if np.isnan(input_array).any():
        for i in range(len(feature_names)):
            if np.isnan(input_array[0, i]):
                input_array[0, i] = 0 # Default to 0 after scaling (which is the median)
    
    probability = float(model.predict_proba(input_array)[0, 1])
    
    print(f"DEBUG: Applicant {req.app_id} Prob: {probability:.4f} | Non-NaN Feats: {np.sum(~np.isnan(features_vector))}/125")

    # 7. SHAP Explainability
    # For Random Forest, TreeExplainer is efficient
    shap_vals = explainer.shap_values(input_array)
    
    # Standardize SHAP output to (n_features,) for the positive class
    if isinstance(shap_vals, list):
        # Format: List[Array(samples, features)] - Take index 1 for Default
        final_shap = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        # Format: Array(samples, features, classes) - Take class 1
        final_shap = shap_vals[0, :, 1]
    else:
        # Format: Array(samples, features) - Assume it's already for the predicted/positive class
        final_shap = shap_vals[0]

    # Pair feature names with their shap values
    feature_impacts = []
    for i, val in enumerate(final_shap):
        if i < len(feature_names):
            feat_name = feature_names[i]
            # Convert to standard float for JSON
            s_val = float(val)
            feature_impacts.append({
                "feature": feat_name,
                "shap_value": s_val,
                "absolute_impact": abs(s_val),
                "effect": "increases_risk" if s_val > 0 else "decreases_risk"
            })
    
    print(f"DEBUG: Calculated {len(feature_impacts)} SHAP impacts. Top 3: {[(x['feature'], round(x['shap_value'], 4)) for x in sorted(feature_impacts, key=lambda x: x['absolute_impact'], reverse=True)[:3]]}")
            
    # Sort by absolute impact and take top 6
    feature_impacts.sort(key=lambda x: x["absolute_impact"], reverse=True)
    top_factors = feature_impacts[:6]
    
    # Random Forest optimal threshold is 0.40
    threshold = OPTIMAL_THRESHOLD
    is_default = probability > threshold
    
    # Save to History
    save_prediction(req.app_id, probability, is_default, req.AMT_INCOME_TOTAL, req.AMT_CREDIT)

    return {
        "probability": probability,
        "is_default": is_default,
        "threshold": threshold,
        "risk_level": "High Risk" if is_default else "Low Risk",
        "explanations": top_factors
    }

@app.get("/history")
async def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 50")
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
