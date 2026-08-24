# requires: pandas, numpy, scikit-learn, joblib
import json

import joblib
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
MODEL_PATH = "models/final_model.joblib"
META_PATH = "models/final_model_meta.json"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 최종 모델: distance_km에 log1p 전환만 적용 (지역 그룹화는 하지 않고 원본 state 사용)
train_df["distance_km_log"] = np.log1p(train_df["distance_km"])
test_df["distance_km_log"] = np.log1p(test_df["distance_km"])

numeric_cols = [
    "distance_km_log", "item_count", "seller_count", "total_price_brl",
    "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour",
]
categorical_cols = ["customer_state", "seller_state", "primary_category", "purchase_weekday"]
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

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regression", LinearRegression()),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"최종 모델(로그거리 적용, 원본 state) 테스트 MAE: {mae:.3f}")
print(f"최종 모델 테스트 RMSE: {rmse:.3f}")
print(f"최종 모델 테스트 R^2: {r2:.3f}")

residuals = y_test.values - y_pred
q10, q90 = np.percentile(residuals, [10, 90])
print(f"잔차(실제-예측) 10%~90% 분위: {q10:.3f} ~ {q90:.3f}")

joblib.dump(pipeline, MODEL_PATH)

# distance_km 원본 결측 대체용 중앙값(로그 전환 전 raw 값 기준, 앱 입력단에서 필요)
distance_km_median_raw = train_df["distance_km"].median()

meta = {
    "raw_numeric_cols": ["distance_km", "item_count", "seller_count", "total_price_brl",
                          "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour"],
    "categorical_cols": categorical_cols,
    "customer_states": sorted(train_df["customer_state"].dropna().unique().tolist()),
    "seller_states": sorted(train_df["seller_state"].dropna().unique().tolist()),
    "primary_categories": sorted(train_df["primary_category"].dropna().unique().tolist()),
    "purchase_weekdays": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "test_mae": mae,
    "test_rmse": rmse,
    "test_r2": r2,
    "residual_q10": float(q10),
    "residual_q90": float(q90),
    "distance_km_median_train": float(distance_km_median_raw),
    "numeric_medians_train": {
        c: float(train_df[c].median()) for c in
        ["item_count", "seller_count", "total_price_brl", "total_freight_brl",
         "total_weight_kg", "total_volume_l", "purchase_hour"]
    },
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("모델 저장 완료:", MODEL_PATH)
print("메타 정보 저장 완료:", META_PATH)
