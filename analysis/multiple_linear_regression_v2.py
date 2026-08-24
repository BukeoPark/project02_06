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
RARE_STATE_RATIO = 0.01  # train 전체 대비 이 비율 미만이면 희소 지역으로 간주

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def group_rare_states(train_col, test_col, ratio):
    counts = train_col.value_counts(dropna=True)
    threshold = ratio * len(train_col)
    rare_states = counts[counts < threshold].index.tolist()
    kept_states = counts[counts >= threshold].index.tolist()
    train_grouped = train_col.apply(lambda x: "OTHER" if x in rare_states else x)
    test_grouped = test_col.apply(lambda x: "OTHER" if (pd.notna(x) and x not in kept_states) else x)
    return train_grouped, test_grouped, rare_states, kept_states


# 1) distance_km 로그 전환 (log1p, 결측은 이후 파이프라인에서 train 중앙값으로 대체)
train_df["distance_km_log"] = np.log1p(train_df["distance_km"])
test_df["distance_km_log"] = np.log1p(test_df["distance_km"])

# 2) 희소 지역 묶기: train 기준 비중 1% 미만인 주(state)는 OTHER로 통합
train_df["customer_state_grouped"], test_df["customer_state_grouped"], rare_cust, kept_cust = group_rare_states(
    train_df["customer_state"], test_df["customer_state"], RARE_STATE_RATIO
)
train_df["seller_state_grouped"], test_df["seller_state_grouped"], rare_seller, kept_seller = group_rare_states(
    train_df["seller_state"], test_df["seller_state"], RARE_STATE_RATIO
)

print(f"customer_state: 희소 지역(train 비중 <{RARE_STATE_RATIO:.0%})으로 OTHER 처리된 주 {len(rare_cust)}개 -> {rare_cust}")
print(f"customer_state: 유지된 주 {len(kept_cust)}개 -> {kept_cust}")
print(f"seller_state: 희소 지역(train 비중 <{RARE_STATE_RATIO:.0%})으로 OTHER 처리된 주 {len(rare_seller)}개 -> {rare_seller}")
print(f"seller_state: 유지된 주 {len(kept_seller)}개 -> {kept_seller}")
print()

numeric_cols = [
    "distance_km_log", "item_count", "seller_count", "total_price_brl",
    "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour",
]
categorical_cols = ["customer_state_grouped", "seller_state_grouped", "primary_category", "purchase_weekday"]
feature_cols = numeric_cols + categorical_cols

X_train = train_df[feature_cols]
y_train = train_df["delivery_days"]
X_test = test_df[feature_cols]
y_test = test_df["delivery_days"]

preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

model_v2 = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regression", LinearRegression()),
])
model_v2.fit(X_train, y_train)
pred_v2 = model_v2.predict(X_test)
mae_v2, rmse_v2, r2_v2 = evaluate(y_test, pred_v2)

# 비교용: 이전 단계(원본 distance_km, 지역 그룹화 없음) 다중회귀 재계산
numeric_cols_v1 = [
    "distance_km", "item_count", "seller_count", "total_price_brl",
    "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour",
]
categorical_cols_v1 = ["customer_state", "seller_state", "primary_category", "purchase_weekday"]
X_train_v1 = train_df[numeric_cols_v1 + categorical_cols_v1]
X_test_v1 = test_df[numeric_cols_v1 + categorical_cols_v1]

preprocessor_v1 = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numeric_cols_v1),
    ("cat", Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols_v1),
])
model_v1 = Pipeline(steps=[
    ("preprocess", preprocessor_v1),
    ("regression", LinearRegression()),
])
model_v1.fit(X_train_v1, y_train)
pred_v1 = model_v1.predict(X_test_v1)
mae_v1, rmse_v1, r2_v1 = evaluate(y_test, pred_v1)

result = pd.DataFrame({
    "모델": ["다중회귀(원본, 12개 변수)", "다중회귀(로그거리+지역그룹, 12개 변수)"],
    "MAE": [mae_v1, mae_v2],
    "RMSE": [rmse_v1, rmse_v2],
    "R^2": [r2_v1, r2_v2],
})
print(result.round(3).to_string(index=False))
