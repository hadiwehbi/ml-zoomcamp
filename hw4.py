import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

path = "/mnt/data/course_lead_scoring_2.csv"
df = pd.read_csv(path)

numerical = df.select_dtypes(include=[np.number]).columns.tolist()
categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()

for c in categorical:
    df[c] = df[c].fillna("NA")
for c in numerical:
    df[c] = df[c].fillna(0.0)
target = "converted"

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1)

y_train = df_train[target].values
y_val = df_val[target].values

# --------------- Q1: ROC AUC for numerical features --------------
num_features = ['lead_score', 'number_of_courses_viewed',
                'interaction_count', 'annual_income']
auc_scores = {}

for f in num_features:
    score = roc_auc_score(y_train, df_train[f])
    if score < 0.5:
        score = roc_auc_score(y_train, -df_train[f])
    auc_scores[f] = round(score, 3)

best_feature = max(auc_scores, key=auc_scores.get)

# ------------ Q2: Train Logistic Regression with DictVectorizer --------------
dv = DictVectorizer(sparse=False)
train_dicts = df_train.drop(columns=[target]).to_dict(orient='records')
val_dicts = df_val.drop(columns=[target]).to_dict(orient='records')

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

model = LogisticRegression(solver='liblinear', C=1.0,
                           max_iter=1000, random_state=1)
model.fit(X_train, y_train)
y_pred_val = model.predict_proba(X_val)[:, 1]

auc_val = round(roc_auc_score(y_val, y_pred_val), 3)

# ------------- Q3: Precision and Recall ------------
thresholds = np.arange(0.0, 1.01, 0.01)
precisions = []
recalls = []

for t in thresholds:
    y_pred_bin = (y_pred_val >= t)
    precisions.append(precision_score(y_val, y_pred_bin))
    recalls.append(recall_score(y_val, y_pred_bin))

diffs = np.abs(np.array(precisions) - np.array(recalls))
intersect_idx = np.argmin(diffs)
intersect_threshold = round(thresholds[intersect_idx], 3)

# ---------------- Q4: F1 score ----------------
f1_scores = 2 * (np.array(precisions) * np.array(recalls)) / \
    (np.array(precisions) + np.array(recalls))
best_f1_idx = np.nanargmax(f1_scores)
best_f1_threshold = round(thresholds[best_f1_idx], 3)

# ------------- Q5: 5-Fold CV ---------------
df_full_train = df_full_train.reset_index(drop=True)
y_full = df_full_train[target].values

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
auc_scores_folds = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_tr = df_full_train.iloc[train_idx]
    df_va = df_full_train.iloc[val_idx]
    y_tr = df_tr[target].values
    y_va = df_va[target].values

    dv_fold = DictVectorizer(sparse=False)
    X_tr = dv_fold.fit_transform(df_tr.drop(
        columns=[target]).to_dict(orient='records'))
    X_va = dv_fold.transform(df_va.drop(
        columns=[target]).to_dict(orient='records'))

    model_fold = LogisticRegression(
        solver='liblinear', C=1.0, max_iter=1000, random_state=1)
    model_fold.fit(X_tr, y_tr)
    y_pred = model_fold.predict_proba(X_va)[:, 1]
    auc_scores_folds.append(roc_auc_score(y_va, y_pred))

std_auc_folds = round(np.std(auc_scores_folds), 3)

# ---------------- Q6: Hyperparameter Tuning ---------------
C_values = [0.000001, 0.001, 1]
cv_results = {}

for C in C_values:
    aucs = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_idx, val_idx in kf.split(df_full_train):
        df_tr = df_full_train.iloc[train_idx]
        df_va = df_full_train.iloc[val_idx]
        y_tr = df_tr[target].values
        y_va = df_va[target].values

        dv_fold = DictVectorizer(sparse=False)
        X_tr = dv_fold.fit_transform(df_tr.drop(
            columns=[target]).to_dict(orient='records'))
        X_va = dv_fold.transform(df_va.drop(
            columns=[target]).to_dict(orient='records'))

        model_cv = LogisticRegression(
            solver='liblinear', C=C, max_iter=1000, random_state=1)
        model_cv.fit(X_tr, y_tr)
        y_pred = model_cv.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, y_pred))

    mean_auc = round(np.mean(aucs), 3)
    std_auc = round(np.std(aucs), 3)
    cv_results[C] = (mean_auc, std_auc)

best_C = sorted(cv_results.items(),
                key=lambda x: (-x[1][0], x[1][1], x[0]))[0][0]
