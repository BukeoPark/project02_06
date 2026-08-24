# requires: pandas, numpy, scikit-learn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

numeric_cols = [
    "distance_km", "item_count", "seller_count", "total_price_brl",
    "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour",
]
categorical_cols = ["customer_state", "seller_state", "primary_category", "purchase_weekday"]
feature_cols = numeric_cols + categorical_cols

X_train = train_df[feature_cols]
y_train = train_df["delivery_days"]
X_test = test_df[feature_cols]
y_test = test_df["delivery_days"]


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# 1) 기준 모델: train 평균으로만 예측
baseline_pred_value = y_train.mean()
baseline_pred = np.full(len(y_test), baseline_pred_value)
baseline_mae, baseline_rmse, baseline_r2 = evaluate(y_test, baseline_pred)

# 2) 단순회귀: distance_km 하나 (결측은 train 중앙값 대체)
distance_median = X_train["distance_km"].median()
simple_train = X_train[["distance_km"]].fillna(distance_median)
simple_test = X_test[["distance_km"]].fillna(distance_median)
simple_model = LinearRegression()
simple_model.fit(simple_train, y_train)
simple_pred = simple_model.predict(simple_test)
simple_mae, simple_rmse, simple_r2 = evaluate(y_test, simple_pred)

# 3) 다중회귀: 수치 8개 + 범주 4개 = 12개 변수
preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

multi_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regression", LinearRegression()),
])
multi_model.fit(X_train, y_train)
multi_pred = multi_model.predict(X_test)
multi_mae, multi_rmse, multi_r2 = evaluate(y_test, multi_pred)

n_num_missing_train = X_train[numeric_cols].isna().sum().sum()
n_cat_missing_train = X_train[categorical_cols].isna().sum().sum()
print(f"수치형 결측치(train, 8개 열 합계): {n_num_missing_train}건 -> 중앙값 대체")
print(f"범주형 결측치(train, 4개 열 합계): {n_cat_missing_train}건 -> 최빈값 대체")
n_ohe_cols = multi_model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols).shape[0]
print(f"원-핫 인코딩 후 범주형 파생 열 수: {n_ohe_cols}개 (수치 8개 + 범주 파생 {n_ohe_cols}개 = 총 입력 {8 + n_ohe_cols}개 열, 원래 변수 12개)")
print()

result = pd.DataFrame({
    "모델": ["기준 모델(평균)", "단순회귀(distance_km)", "다중회귀(12개 변수)"],
    "MAE": [baseline_mae, simple_mae, multi_mae],
    "RMSE": [baseline_rmse, simple_rmse, multi_rmse],
    "R^2": [baseline_r2, simple_r2, multi_r2],
})
print(result.round(3).to_string(index=False))
