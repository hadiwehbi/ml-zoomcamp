import os
import subprocess

from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import load_model, predict_proba


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "data/diabetes.csv")

app = FastAPI(title="Diabetes Prediction API", version="1.0")


class Patient(BaseModel):
    Pregnancies: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)


def ensure_model_exists():
    if os.path.exists(MODEL_PATH):
        return

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}' and dataset not found at '{DATA_PATH}'. "
            "Place dataset at data/diabetes.csv or set DATA_PATH env var."
        )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    subprocess.check_call(
        ["python", "train.py", "--data-path", DATA_PATH, "--model-path", MODEL_PATH]
    )


@app.on_event("startup")
def startup_event():
    ensure_model_exists()
    app.state.model = load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    features = patient.model_dump()
    proba = predict_proba(app.state.model, features)
    return {"diabetes_probability": proba, "diabetes": int(proba >= 0.5)}
