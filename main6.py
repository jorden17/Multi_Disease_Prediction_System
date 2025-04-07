from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scalers
try:
    diabetes_model = joblib.load("models/diabetes_model.sav")
    diabetes_scaler = joblib.load("models/scaler_diabetes.sav")
    heart_model = joblib.load("models/heart_disease_model.sav")
    heart_scaler = joblib.load("models/scaler_heart.sav")
    parkinson_model = joblib.load("models/parkinsons_model.sav")
    parkinson_scaler = joblib.load("models/scaler_parkinsons.sav")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# Input Schema
class DiabetesInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    pedigree: float
    age: float

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class ParkinsonInput(BaseModel):
    fo: float
    fhi: float
    flo: float
    jitter_percent: float
    jitter_abs: float
    rap: float
    ppq: float
    ddp: float
    shimmer: float
    shimmer_db: float
    apq3: float
    apq5: float
    apq: float
    dda: float
    nhr: float
    hnr: float
    rpde: float
    dfa: float
    spread1: float
    spread2: float
    d2: float
    ppe: float

# Prediction Endpoint
@app.post("/predict/diabetes")
async def predict_diabetes(data: DiabetesInput):
    try:
        # Prepare input data for the model
        input_data = np.array([[getattr(data, field) for field in data.__annotations__.keys()]])
        input_data_scaled = diabetes_scaler.transform(input_data)  # Scale the input data
        prediction = diabetes_model.predict(input_data_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in diabetes prediction: {str(e)}")
    
@app.post("/predict/heart")
async def predict_heart(data: HeartInput):
    try:
        # Prepare input data for the model
        input_data = np.array([[getattr(data, field) for field in data.__annotations__.keys()]])
        input_data_scaled = heart_scaler.transform(input_data)  # Scale the input data
        prediction = heart_model.predict(input_data_scaled)[0]
        probability = heart_model.predict_proba(input_data_scaled)[0][1]  # Probability of heart disease
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return {"prediction": result, "probability": f"{probability * 100:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in heart disease prediction: {str(e)}")

@app.post("/predict/parkinson")
async def predict_parkinson(data: ParkinsonInput):
    try:
        # Prepare input data for the model
        input_data = np.array([[getattr(data, field) for field in data.__annotations__.keys()]])
        input_data_scaled = parkinson_scaler.transform(input_data)  # Scale the input data
        prediction = parkinson_model.predict(input_data_scaled)[0]
        probability = parkinson_model.predict_proba(input_data_scaled)[0][1]  # Probability of Parkinson's
        result = "Parkinson’s Detected" if prediction == 1 else "No Parkinson’s"
        return {"prediction": result, "probability": f"{probability * 100:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Parkinson's prediction: {str(e)}")

# Serve frontend HTML and static assets
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")