import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("/mnt/data/car_fuel_efficiency.csv")
FEATURES = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
TARGET = 'fuel_efficiency_mpg'

def split_data(data: pd.DataFrame, seed: int):
    """Shuffle and split into 60/20/20 train/val/test, deterministic by seed."""
    n = len(data)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    # n_test equals the remainder to avoid rounding issues
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train+n_val]
    idx_test = idx[n_train+n_val:]
    return (data.iloc[idx_train].reset_index(drop=True),
            data.iloc[idx_val].reset_index(drop=True),
            data.iloc[idx_test].reset_index(drop=True))

def train_linear_regression(X, y, r: float = 0.0):
    """Closed-form linear regression; L2-regularized if r>0 (bias not regularized)."""
    X = np.column_stack([np.ones(len(X)), X])
    XTX = X.T.dot(X)
    if r > 0:
        reg = r * np.eye(XTX.shape[0])
        reg[0, 0] = 0  # don't regularize bias
        XTX = XTX + reg
    w = np.linalg.inv(XTX).dot(X.T).dot(y)
    return w

def predict(w, X):
    X = np.column_stack([np.ones(len(X)), X])
    return X.dot(w)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def header(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

df = pd.read_csv(DATA_PATH)
df = df[FEATURES + [TARGET]].copy()

# header("EDA: Target distribution (skewness)")
# print("Skewness of fuel_efficiency_mpg:", float(df[TARGET].skew()))

# Q1: Column with missing values
header("Q1: Column with missing values")
missing_counts = df.isna().sum()
cols_with_na = missing_counts[missing_counts > 0].sort_values(ascending=False)
print("Missing counts by column:")
print(cols_with_na.to_string())
q1_answer = cols_with_na.index.tolist()
print("Answer (column(s) with NAs):", q1_answer)

# Q2: Median of horsepower
header("Q2: Median horsepower")
median_hp = float(df['horsepower'].median())
print("Median horsepower:", median_hp)

header("Prepare & Split (seed=42)")
df_train, df_val, df_test = split_data(df, seed=42)
print(f"Split sizes -> train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")


# Q3: Missing value strategies (fill 0 vs mean), no regularization
header("Q3: Imputation comparison (0 vs mean), no regularization")

# Option A: fill with 0
train_zero = df_train.copy()
val_zero = df_val.copy()
for c in FEATURES:
    train_zero[c] = train_zero[c].fillna(0)
    val_zero[c] = val_zero[c].fillna(0)

X_train_zero = train_zero[FEATURES].values
y_train = train_zero[TARGET].values
w_zero = train_linear_regression(X_train_zero, y_train, r=0.0)

X_val_zero = val_zero[FEATURES].values
y_val = val_zero[TARGET].values
rmse_zero = rmse(y_val, predict(w_zero, X_val_zero))

# Option B: fill with mean (from training only)
train_mean = df_train.copy()
val_mean = df_val.copy()
means = train_mean[FEATURES].mean()
for c in FEATURES:
    train_mean[c] = train_mean[c].fillna(means[c])
    val_mean[c] = val_mean[c].fillna(means[c])

X_train_mean = train_mean[FEATURES].values
w_mean = train_linear_regression(X_train_mean, y_train, r=0.0)

X_val_mean = val_mean[FEATURES].values
rmse_mean = rmse(y_val, predict(w_mean, X_val_mean))

print(f"RMSE (fill 0):   {round(rmse_zero, 2)}")
print(f"RMSE (fill mean): {round(rmse_mean, 2)}")

if round(rmse_zero, 2) < round(rmse_mean, 2):
    print("Better option: With 0")
elif round(rmse_zero, 2) > round(rmse_mean, 2):
    print("Better option: With mean")
else:
    print("Better option: Both are equally good")


# Q4: Regularized linear regression, fill NAs with 0
header("Q4: Ridge (L2) with different r's (fill NAs=0)")
r_list = [0, 0.01, 0.1, 1, 5, 10, 100]
val_scores = {}

for r in r_list:
    w_r = train_linear_regression(X_train_zero, y_train, r=r)
    score = rmse(y_val, predict(w_r, X_val_zero))
    val_scores[r] = round(score, 2)

for r in r_list:
    print(f"r={r:>6}: RMSE={val_scores[r]}")

best_rmse = min(val_scores.values())
best_r = min([r for r, s in val_scores.items() if s == best_rmse])
print(f"Best r (tie -> smallest): {best_r} with RMSE={best_rmse}")


# Q5: Sensitivity to seed (0..9), fill 0, no regularization
header("Q5: Std of val RMSE over seeds 0..9 (fill 0, no reg)")
seed_scores = []
for seed in range(10):
    tr, va, te = split_data(df, seed=seed)
    tr = tr.copy(); va = va.copy()
    for c in FEATURES:
        tr[c] = tr[c].fillna(0)
        va[c] = va[c].fillna(0)
    Xtr = tr[FEATURES].values
    ytr = tr[TARGET].values
    Xva = va[FEATURES].values
    yva = va[TARGET].values
    w = train_linear_regression(Xtr, ytr, r=0.0)
    score = rmse(yva, predict(w, Xva))
    seed_scores.append(score)
    print(f"Seed {seed}: RMSE={score:.6f}")

std_scores = float(np.std(seed_scores))
print("Std over seeds:", round(std_scores, 3))


# Q6: Seed=9; train on train+val; r=0.001; test RMSE
header("Q6: Final model (seed=9, train+val, r=0.001, fill 0)")
tr, va, te = split_data(df, seed=9)
comb = pd.concat([tr, va], ignore_index=True)

for c in FEATURES:
    comb[c] = comb[c].fillna(0)
    te[c] = te[c].fillna(0)

X_comb = comb[FEATURES].values
y_comb = comb[TARGET].values
w_q6 = train_linear_regression(X_comb, y_comb, r=0.001)

X_te = te[FEATURES].values
y_te = te[TARGET].values
rmse_q6 = rmse(y_te, predict(w_q6, X_te))

print("Test RMSE (seed=9, r=0.001):", rmse_q6, "-> rounded:", round(rmse_q6, 3))
