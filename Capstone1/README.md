# Diabetes Prediction (Capstone 1)

This project builds and deploys a machine learning model that predicts whether a patient is likely to have diabetes based on diagnostic measurements.

## Problem statement

Diabetes is a chronic disease where early risk detection can help reduce complications.  
The goal of this project is to train a binary classification model that predicts **Outcome** (0/1) from patient measurements (e.g., glucose, BMI, age).

## Dataset

Input file: `data/diabetes.csv`  
Expected columns (Pima-style dataset):

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target)

> Note: In this dataset, zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are typically invalid and are treated as missing values.

## Project structure

```
.
├── README.md
├── notebook.ipynb
├── train.py
├── predict.py
├── serve.py
├── requirements.txt
├── Dockerfile
├── data/
│   └── diabetes.csv
└── models/
    └── model.pkl
```

## How to run locally

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train the model

Put the dataset at `data/diabetes.csv`, then:

```bash
python train.py --data-path data/diabetes.csv --model-path models/model.pkl
```

### 3) Start the API

```bash
uvicorn serve:app --host 0.0.0.0 --port 9696
```

### 4) Test the API

Health check:

```bash
curl -s http://localhost:9696/health
```

Prediction:

```bash
curl -s -X POST http://localhost:9696/predict   -H "Content-Type: application/json"   -d '{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28.3,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 33
  }'
```

## Docker

### Build

```bash
docker build -t diabetes-capstone1 .
```

### Run

```bash
docker run --rm -p 9696:9696 diabetes-capstone1
```

Then repeat the `curl` requests above.

## Notes / Limitations

- This model is trained on a specific public dataset and should **not** be used for clinical decision-making.
- Performance depends on data quality; missing/invalid values are imputed.
