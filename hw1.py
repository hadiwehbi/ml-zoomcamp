import sys, io, numpy as np, pandas as pd

print("# Q1. Pandas version")
print(pd.__version__)
print()

# Getting the data
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
df = pd.read_csv(url)

# Q2. Records count
print("# Q2. Records count")
print(len(df))
print()

# Q3. Fuel types
fuel_col = "fuel_type" if "fuel_type" in df.columns else None
if fuel_col is None:
    print("# Could not find 'fuel_type' column. Columns are:", list(df.columns))
else:
    print("# Q3. Fuel types")
    print(df[fuel_col].nunique(dropna=True))
print()

# Q4. Missing values
print("# Q4. Columns with missing values")
print((df.isna().sum() > 0).sum())
print()

# Q5. Max fuel efficiency of cars from Asia
origin_col = "origin"
fe_col = "fuel_efficiency" if "fuel_efficiency" in df.columns else None

if fe_col is None:
    for cand in ["mpg", "fuel_eff", "fuel_economy", "combined_mpg"]:
        if cand in df.columns:
            fe_col = cand
            break

if fe_col is None or origin_col not in df.columns:
    print("# Q5. Couldn't identify needed columns. Columns are:", list(df.columns))
else:
    max_asia = df.loc[df[origin_col] == "Asia", fe_col].max()
    print("# Q5. Max fuel efficiency (Asia)")
    # Print with two decimals to match choices like 13.75 / 23.75 etc.
    print(f"{float(max_asia):.2f}")
print()

# Q6. Median value of horsepower before/after fillna with most frequent
hp_col = "horsepower" if "horsepower" in df.columns else None
if hp_col is None:
    print("# Q6. Column 'horsepower' not found. Columns are:", list(df.columns))
else:
    med_before = df[hp_col].median()
    mode_vals = df[hp_col].mode(dropna=True)
    most_freq = mode_vals.iloc[0] if not mode_vals.empty else None

    df2 = df.copy()
    if most_freq is not None:
        df2[hp_col] = df2[hp_col].fillna(most_freq)
    med_after = df2[hp_col].median()

    print("# Q6. Median horsepower before fillna:", med_before)
    print("# Q6. Most frequent horsepower:", most_freq)
    print("# Q6. Median horsepower after fillna:", med_after)
    if med_after > med_before:
        print("Changed? Yes, it increased")
    elif med_after < med_before:
        print("Changed? Yes, it decreased")
    else:
        print("Changed? No")
print()

# Q7. Sum of weights via linear algebra steps
vw_col = "vehicle_weight" if "vehicle_weight" in df.columns else None
my_col = "model_year" if "model_year" in df.columns else None

if origin_col in df.columns and vw_col and my_col:
    asia = df[df[origin_col] == "Asia"][[vw_col, my_col]].head(7)
    if len(asia) < 7:
        print("# Q7. Not enough Asia rows; found:", len(asia))
    else:
        X = asia.to_numpy()
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX)

        y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200], dtype=float)

        w = XTX_inv @ X.T @ y
        s = w.sum()
        print("# Q7. Sum of all elements of w")
        print("raw:", s)
        print("rounded(3):", round(float(s), 3))
else:
    print("# Q7. Needed columns not found. Columns are:", list(df.columns))
