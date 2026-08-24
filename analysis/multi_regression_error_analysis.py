# requires: pandas, numpy, scikit-learn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
RARE_STATE_RATIO = 0.01

STATE_TO_REGION = {
    "AC": "북부", "AP": "북부", "AM": "북부", "PA": "북부", "RO": "북부", "RR": "북부", "TO": "북부",
    "AL": "동북부", "BA": "동북부", "CE": "동북부", "MA": "동북부", "PB": "동북부",
    "PE": "동북부", "PI": "동북부", "RN": "동북부", "SE": "동북부",
    "DF": "중서부", "GO": "중서부", "MT": "중서부", "MS": "중서부",
    "ES": "동남부", "MG": "동남부", "RJ": "동남부", "SP": "동남부",
    "PR": "남부", "RS": "남부", "SC": "남부",
}

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["distance_km_log"] = np.log1p(train_df["distance_km"])
test_df["distance_km_log"] = np.log1p(test_df["distance_km"])


def group_rare(train_col, test_col, ratio):
    counts = train_col.value_counts(dropna=True)
    threshold = ratio * len(train_col)
    kept = counts[counts >= threshold].index.tolist()
    train_g = train_col.apply(lambda x: "OTHER" if x not in kept else x)
    test_g = test_col.apply(lambda x: "OTHER" if (pd.notna(x) and x not in kept) else x)
    return train_g, test_g


train_df["customer_state_grouped"], test_df["customer_state_grouped"] = group_rare(
    train_df["customer_state"], test_df["customer_state"], RARE_STATE_RATIO
)
train_df["seller_state_grouped"], test_df["seller_state_grouped"] = group_rare(
    train_df["seller_state"], test_df["seller_state"], RARE_STATE_RATIO
)

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
model = Pipeline(steps=[("preprocess", preprocessor), ("regression", LinearRegression())])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

overall_mae = mean_absolute_error(y_test, y_pred)
print(f"다중회귀(로그거리+1% 미만 OTHER 묶음) 테스트 전체 MAE: {overall_mae:.3f}")
print()

result_df = test_df.copy()
result_df["predicted_delivery_days"] = y_pred
result_df["abs_error"] = (result_df["delivery_days"] - result_df["predicted_delivery_days"]).abs()

# 1) 절대 오차가 큰 주문 20건
top20 = result_df.sort_values("abs_error", ascending=False).head(20)
cols_to_show = ["order_id", "delivery_days", "predicted_delivery_days", "abs_error",
                 "distance_km", "customer_state", "seller_state", "primary_category"]
top20_display = top20[cols_to_show].copy()
top20_display["delivery_days"] = top20_display["delivery_days"].round(2)
top20_display["predicted_delivery_days"] = top20_display["predicted_delivery_days"].round(2)
top20_display["abs_error"] = top20_display["abs_error"].round(2)
print("절대 오차 상위 20건:")
print(top20_display.to_string(index=False))
print()

# 2) 고객 지역(customer_region)별 데이터수 & MAE
result_df["customer_region"] = result_df["customer_state"].map(STATE_TO_REGION)
region_summary = result_df.groupby("customer_region", dropna=False).agg(
    n=("abs_error", "size"),
    mae=("abs_error", "mean"),
).reset_index().sort_values("n", ascending=False)
region_summary["mae"] = region_summary["mae"].round(3)
print("고객 지역(customer_region)별 데이터수 & MAE:")
print(region_summary.to_string(index=False))
print()

# 3) 거리 구간별 데이터수 & MAE
bins = [-0.001, 200, 500, 1000, 2000, np.inf]
labels = ["0-200km", "200-500km", "500-1000km", "1000-2000km", "2000km+"]
result_df["distance_bin"] = pd.cut(result_df["distance_km"], bins=bins, labels=labels)
distance_summary = result_df.groupby("distance_bin", observed=True).agg(
    n=("abs_error", "size"),
    mae=("abs_error", "mean"),
).reset_index()
distance_summary["mae"] = distance_summary["mae"].round(3)
print("거리 구간별 데이터수 & MAE:")
print(distance_summary.to_string(index=False))
