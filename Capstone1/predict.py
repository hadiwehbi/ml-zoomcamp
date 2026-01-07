import pickle
from typing import Dict, Any

import pandas as pd


def load_model(model_path: str = "models/model.pkl"):
    with open(model_path, "rb") as f_in:
        model = pickle.load(f_in)
    return model


def predict_proba(model, patient_features: Dict[str, Any]) -> float:
    df = pd.DataFrame([patient_features])
    proba = model.predict_proba(df)[:, 1][0]
    return float(proba)
