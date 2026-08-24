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

# 브라질 공식 5개 지역권 (고정된 지리적 사실 -> train/test 동일 적용해도 정보 누출 아님)
STATE_TO_REGION = {
    "AC": "북부", "AP": "북부", "AM": "북부", "PA": "북부", "RO": "북부", "RR": "북부", "TO": "북부",
    "AL": "동북부", "BA": "동북부", "CE": "동북부", "MA": "동북부", "PB": "동북부",
    "PE": "동북부", "PI": "동북부", "RN": "동북부", "SE": "동북부",
    "DF": "중서부", "GO": "중서부", "MT": "중서부", "MS": "중서부",
    "ES": "동남부", "MG": "동남부", "RJ": "동남부", "SP": "동남부",
    "PR": "남부", "RS": "남부", "SC": "남부",
}


def to_region(col):
    return col.map(STATE_TO_REGION)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def fit_and_eval(train_df, test_df, numeric_cols, categorical_cols):
    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df["delivery_days"]
    X_test = test_df[numeric_cols + categorical_cols]
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
    pred = model.predict(X_test)
    return evaluate(y_test, pred)


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["distance_km_log"] = np.log1p(train_df["distance_km"])
test_df["distance_km_log"] = np.log1p(test_df["distance_km"])

train_df["customer_region"] = to_region(train_df["customer_state"])
test_df["customer_region"] = to_region(test_df["customer_state"])
train_df["seller_region"] = to_region(train_df["seller_state"])
test_df["seller_region"] = to_region(test_df["seller_state"])

print("customer_region 분포 (train):")
print(train_df["customer_region"].value_counts())
print()
print("seller_region 분포 (train):")
print(train_df["seller_region"].value_counts())
print()

numeric_cols = [
    "distance_km_log", "item_count", "seller_count", "total_price_brl",
    "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour",
]

# v1: 원본 state, 원본 distance_km
mae_v1, rmse_v1, r2_v1 = fit_and_eval(
    train_df, test_df,
    ["distance_km", "item_count", "seller_count", "total_price_brl",
     "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour"],
    ["customer_state", "seller_state", "primary_category", "purchase_weekday"],
)

# v2: 로그거리 + train 비중 1% 미만 OTHER 묶음(이전 단계 결과 재계산)
RARE_STATE_RATIO = 0.01


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
mae_v2, rmse_v2, r2_v2 = fit_and_eval(
    train_df, test_df,
    numeric_cols,
    ["customer_state_grouped", "seller_state_grouped", "primary_category", "purchase_weekday"],
)

# v3: 로그거리 + 지역권(customer_region, seller_region)
mae_v3, rmse_v3, r2_v3 = fit_and_eval(
    train_df, test_df,
    numeric_cols,
    ["customer_region", "seller_region", "primary_category", "purchase_weekday"],
)

result = pd.DataFrame({
    "모델": [
        "다중회귀(원본, 12개 변수)",
        "다중회귀(로그거리+1% 미만 OTHER 묶음)",
        "다중회귀(로그거리+지역권 재분류)",
    ],
    "MAE": [mae_v1, mae_v2, mae_v3],
    "RMSE": [rmse_v1, rmse_v2, rmse_v3],
    "R^2": [r2_v1, r2_v2, r2_v3],
})
print(result.round(3).to_string(index=False))
