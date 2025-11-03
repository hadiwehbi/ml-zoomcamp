import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

df = pd.read_csv("car_fuel_efficiency.csv")
df = df.fillna(0)

target = 'fuel_efficiency_mpg'
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1)

y_train = df_train[target].values
y_val = df_val[target].values

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.drop(
    columns=[target]).to_dict(orient='records'))
X_val = dv.transform(df_val.drop(columns=[target]).to_dict(orient='records'))

# ---------------- Q1 ----------------
dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train)
split_feature = dv.feature_names_[np.argmax(dt.feature_importances_)]
print("Q1. Split feature:", split_feature)

# ---------------- Q2 ----------------
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
rmse_q2 = mean_squared_error(y_val, rf.predict(X_val), squared=False)
print("Q2. RMSE:", round(rmse_q2, 3))

# ---------------- Q3 ----------------
rmse_by_n = {}
for n in range(10, 201, 10):
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    rmse_by_n[n] = mean_squared_error(y_val, rf.predict(X_val), squared=False)

prev_rmse, stop_n = None, None
for n, rmse in rmse_by_n.items():
    if prev_rmse is not None and round(rmse, 3) >= round(prev_rmse, 3):
        stop_n = n
        break
    prev_rmse = rmse
if stop_n is None:
    stop_n = 200
print("Q3. RMSE stops improving after:", stop_n)

# ---------------- Q4 ----------------
depths = [10, 15, 20, 25]
rmse_by_depth = {}
for d in depths:
    rmses = []
    for n in range(10, 201, 10):
        rf = RandomForestRegressor(
            n_estimators=n, max_depth=d, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        rmses.append(mean_squared_error(
            y_val, rf.predict(X_val), squared=False))
    rmse_by_depth[d] = np.mean(rmses)
best_depth = min(rmse_by_depth, key=rmse_by_depth.get)
print("Q4. Best max_depth:", best_depth)

# ---------------- Q5 ----------------
rf = RandomForestRegressor(
    n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
imp = dict(zip(dv.feature_names_, rf.feature_importances_))
candidates = ['vehicle_weight', 'horsepower',
              'acceleration', 'engine_displacement']
important = max(candidates, key=lambda f: sum(
    v for k, v in imp.items() if k.startswith(f)))
print("Q5. Most important feature:", important)

# ---------------- Q6 ----------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'max_depth': 6, 'min_child_weight': 1,
    'objective': 'reg:squarederror', 'nthread': 8,
    'seed': 1, 'verbosity': 0
}

for eta in [0.3, 0.1]:
    params['eta'] = eta
    model = xgb.train(params, dtrain, num_boost_round=100)
    rmse = mean_squared_error(y_val, model.predict(dval), squared=False)
    print(f"Q6. eta={eta}: RMSE={rmse:.3f}")
