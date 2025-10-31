import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import requests

MODEL_FILE = "pipeline_v1.bin"

try:
    with open(MODEL_FILE, "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Model file '{MODEL_FILE}' not found. "
                       f"Download it using:\n"
                       f"wget https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin")

# ==============================
# Q3: Local Scoring Example
# ==============================
def score_client(client):
    """Return probability of conversion for a given client dictionary."""
    return float(pipeline.predict_proba([client])[0, 1])

if __name__ == "__main__":
    # Example 1: Question 3 record
    client_q3 = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }

    prob_q3 = score_client(client_q3)
    print(f"Q3: Probability (paid_ads client) = {prob_q3:.3f}")

    # Example 2: Question 4/6 record
    client_q4 = {
        "lead_source": "organic_search",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0
    }

    prob_q4 = score_client(client_q4)
    print(f"Q4: Probability (organic_search client) = {prob_q4:.3f}")
    print("\nRun the FastAPI app with:\n  uv run uvicorn hw05_solution:app --reload\n")

# ==============================
# Q4–Q6: FastAPI Service
# ==============================
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI(title="Lead Scoring Model API", version="1.0")

@app.get("/")
def root():
    return {"status": "ok", "message": "Lead Scoring API is running."}

@app.post("/predict")
def predict(lead: Lead):
    """POST endpoint to score a lead."""
    data = lead.model_dump()
    proba = score_client(data)
    return {"probability": round(proba, 3)}
