import argparse
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET_COL = "Outcome"


@dataclass
class TrainResult:
    name: str
    auc: float
    pipeline: Pipeline


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")
    return df


def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()

    # Replace invalid zeros with NaN for known columns
    for c in ZERO_AS_MISSING_COLS:
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return X, y, numeric_cols


def build_pipeline(model, numeric_cols: List[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_cols)],
        remainder="drop",
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    return pipe


def evaluate_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int = 42) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Cross-validated predicted probabilities
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    return float(auc)


def train_best_model(data_path: str) -> TrainResult:
    df = load_data(data_path)
    X, y, numeric_cols = make_features(df)

    candidates = [
        ("logreg", LogisticRegression(max_iter=500, solver="lbfgs")),
        ("rf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(random_state=42)),
    ]

    results: List[TrainResult] = []
    for name, model in candidates:
        pipe = build_pipeline(model, numeric_cols)
        auc = evaluate_model(pipe, X, y)
        results.append(TrainResult(name=name, auc=auc, pipeline=pipe))

    best = max(results, key=lambda r: r.auc)

    # Fit best model on full data
    best.pipeline.fit(X, y)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to diabetes.csv")
    parser.add_argument("--model-path", default="models/model.pkl", help="Where to save the trained model")
    args = parser.parse_args()

    best = train_best_model(args.data_path)

    print(f"Best model: {best.name} | CV ROC-AUC: {best.auc:.4f}")

    with open(args.model_path, "wb") as f_out:
        pickle.dump(best.pipeline, f_out)

    print(f"Saved model to: {args.model_path}")


if __name__ == "__main__":
    main()
