# House Price Prediction (Ames Housing)

## Problem

Predict the sale price (`SalePrice`) of houses in Ames, Iowa, using property features such as quality, size, location, and condition.

## Dataset

- Source: Kaggle – _House Prices: Advanced Regression Techniques_
- Target: `SalePrice`
- Rows: ~1,460
- Examples of features (from `data_description.txt`):
  - `OverallQual`: Overall material and finish of the house (1–10)
  - `GrLivArea`: Above-ground living area (sq ft)
  - `GarageCars`: Garage capacity (number of cars)
  - `TotalBsmtSF`: Total basement area (sq ft)
  - `Neighborhood`: Physical location within Ames
  - `HouseStyle`: Dwelling style (1Story, 2Story, etc.)
  - `SaleCondition`: Condition of sale (Normal, Family, Abnorml, etc.)

## EDA Summary

- Prices are right-skewed (most houses in mid range, few very expensive).
- Higher `OverallQual` and larger `GrLivArea` strongly correlate with higher `SalePrice`.
- Some features contain missing values; numeric ones are filled with 0, categoricals with `"Missing"`.

## Modeling

- **Baseline model**: Linear Regression with one-hot encoded features (`DictVectorizer`).
- **Final model**: RandomForestRegressor
  - `n_estimators=200`
  - `max_depth=10`

Final performance (RMSE):

- Validation RMSE (RandomForest): `XXX`
- Test RMSE (RandomForest): `YYY`

_(Fill in the actual numbers from your notebook/train.py output.)_

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```