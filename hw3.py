import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

path = "/mnt/data/course_lead_scoring.csv"
df = pd.read_csv(path)

numerical = df.select_dtypes(include=[np.number]).columns.tolist()
categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()

for c in categorical:
    df[c] = df[c].fillna("NA")
for c in numerical:
    df[c] = df[c].fillna(0.0)

target = "converted"

# Q1
mode_industry = df["industry"].mode()[0]

# Q2
corr = df[numerical].corr()
pairs = {
    "interaction_count and lead_score": corr.loc["interaction_count", "lead_score"],
    "number_of_courses_viewed and lead_score": corr.loc["number_of_courses_viewed", "lead_score"],
    "number_of_courses_viewed and interaction_count": corr.loc["number_of_courses_viewed", "interaction_count"],
    "annual_income and interaction_count": corr.loc["annual_income", "interaction_count"]
}
max_pair = max(pairs, key=lambda x: abs(pairs[x]))

# Split
X = df.drop(columns=[target])
y = df[target]

X_train_full, X_temp, y_train_full, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Q3 - Encode categorical features numerically
X_train_encoded = X_train_full.copy()
for col in X_train_encoded.select_dtypes(exclude=[np.number]).columns:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train_encoded[col])

cat_cols = X_train_full.select_dtypes(exclude=[np.number]).columns.tolist()
mi_scores = mutual_info_classif(
    X_train_encoded[cat_cols], y_train_full, discrete_features=True, random_state=42)
mi_dict = {col: round(score, 2) for col, score in zip(cat_cols, mi_scores)}
best_mi = max(mi_dict, key=mi_dict.get)

# Q4 Logistic regression
X_train_dummies = pd.get_dummies(X_train_full, drop_first=True)
X_val_dummies = pd.get_dummies(X_val, drop_first=True)
X_val_dummies = X_val_dummies.reindex(
    columns=X_train_dummies.columns, fill_value=0)

model = LogisticRegression(solver="liblinear", C=1.0,
                           max_iter=1000, random_state=42)
model.fit(X_train_dummies, y_train_full)
val_acc = round(accuracy_score(y_val, model.predict(X_val_dummies)), 2)

# Q5 Feature elimination
orig_acc = accuracy_score(y_val, model.predict(X_val_dummies))
feature_diffs = {}

for feature in X_train_full.columns:
    X_train_sub = X_train_full.drop(columns=[feature])
    X_val_sub = X_val.drop(columns=[feature])
    X_train_sub_dum = pd.get_dummies(X_train_sub, drop_first=True)
    X_val_sub_dum = pd.get_dummies(X_val_sub, drop_first=True)
    X_val_sub_dum = X_val_sub_dum.reindex(
        columns=X_train_sub_dum.columns, fill_value=0)
    model_sub = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    model_sub.fit(X_train_sub_dum, y_train_full)
    acc_sub = accuracy_score(y_val, model_sub.predict(X_val_sub_dum))
    feature_diffs[feature] = orig_acc - acc_sub

min_diff_feature = min(feature_diffs, key=lambda x: abs(feature_diffs[x]))

# Q6 Regularized logistic regression
Cs = [0.01, 0.1, 1, 10, 100]
val_scores = {}
for C in Cs:
    model_r = LogisticRegression(
        solver="liblinear", C=C, max_iter=1000, random_state=42)
    model_r.fit(X_train_dummies, y_train_full)
    y_pred = model_r.predict(X_val_dummies)
    val_scores[C] = round(accuracy_score(y_val, y_pred), 3)

best_acc = max(val_scores.values())
best_Cs = [C for C, acc in val_scores.items() if acc == best_acc]
best_C = min(best_Cs)
