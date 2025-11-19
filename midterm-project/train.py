import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle


numeric_features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "LotArea",
]

categorical_features = [
    "Neighborhood",
    "HouseStyle",
    "BldgType",
    "Exterior1st",
    "SaleCondition",
]

features = numeric_features + categorical_features

output_file = "model.bin"


def main():
    df = pd.read_csv("data/train.csv")

    # split train/test
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    y_full_train = df_full_train.SalePrice.values
    y_test = df_test.SalePrice.values

    # fill missing values
    for df_part in (df_full_train, df_test):
        df_part[numeric_features] = df_part[numeric_features].fillna(0)
        for col in categorical_features:
            df_part[col] = df_part[col].fillna("Missing")

    df_full_train = df_full_train[features]
    df_test = df_test[features]

    dv = DictVectorizer(sparse=False)

    train_dicts = df_full_train.to_dict(orient="records")
    X_full_train = dv.fit_transform(train_dicts)

    test_dicts = df_test.to_dict(orient="records")
    X_test = dv.transform(test_dicts)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=1,
        n_jobs=-1,
    )

    model.fit(X_full_train, y_full_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE on test: {rmse:.2f}")

    with open(output_file, "wb") as f_out:
        pickle.dump((dv, model), f_out)

    print(f"Model saved to {output_file}")


if __name__ == "__main__":
    main()
