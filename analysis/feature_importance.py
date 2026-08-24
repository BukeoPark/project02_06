# requires: pandas, numpy, scikit-learn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN_PATH = "data/processed/train.csv"
NUMERIC_OUT = "outputs/tables/feature_importance_numeric.csv"
CATEGORICAL_OUT = "outputs/tables/feature_importance_categorical.csv"

train_df = pd.read_csv(TRAIN_PATH)
train_df["distance_km_log"] = np.log1p(train_df["distance_km"])

numeric_cols = [
    "distance_km_log", "item_count", "seller_count", "total_price_brl",
    "total_freight_brl", "total_weight_kg", "total_volume_l", "purchase_hour",
]
categorical_cols = ["customer_state", "seller_state", "primary_category", "purchase_weekday"]

X_train = train_df[numeric_cols + categorical_cols]
y_train = train_df["delivery_days"]

# 최종 모델과 동일한 변수 구성이지만, 계수를 해석 가능하게 만들기 위해
# 수치형은 표준화(StandardScaler), 범주형은 기준범주 제외(drop="first") 인코딩 적용
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first")),
    ]), categorical_cols),
])

model = Pipeline(steps=[("preprocess", preprocessor), ("regression", LinearRegression())])
model.fit(X_train, y_train)

feature_names = model.named_steps["preprocess"].get_feature_names_out()
coefs = model.named_steps["regression"].coef_

coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})

numeric_rows = coef_df[coef_df["feature"].str.startswith("num__")].copy()
numeric_rows["feature"] = numeric_rows["feature"].str.replace("num__", "", regex=False)
numeric_rows = numeric_rows.rename(columns={"coefficient": "coef_per_1sd"})
numeric_rows = numeric_rows.sort_values("coef_per_1sd", key=lambda s: s.abs(), ascending=False)

categorical_rows = coef_df[coef_df["feature"].str.startswith("cat__")].copy()
categorical_rows["feature"] = categorical_rows["feature"].str.replace("cat__", "", regex=False)


def split_var_category(name, cat_cols):
    for c in sorted(cat_cols, key=len, reverse=True):
        if name.startswith(c + "_"):
            return c, name[len(c) + 1:]
    return name, ""


categorical_rows[["variable", "category"]] = categorical_rows["feature"].apply(
    lambda x: pd.Series(split_var_category(x, categorical_cols))
)

reference_rows = []
for i, col in enumerate(categorical_cols):
    encoder = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    ref_category = encoder.categories_[i][encoder.drop_idx_[i]]
    reference_rows.append({"variable": col, "category": ref_category, "coefficient": 0.0, "is_reference": True})

categorical_rows = categorical_rows.rename(columns={"coefficient": "coefficient"})
categorical_rows["is_reference"] = False
categorical_rows = categorical_rows[["variable", "category", "coefficient", "is_reference"]]
categorical_rows = pd.concat([categorical_rows, pd.DataFrame(reference_rows)], ignore_index=True)
categorical_rows = categorical_rows.sort_values(["variable", "coefficient"], ascending=[True, False])

numeric_rows[["feature", "coef_per_1sd"]].to_csv(NUMERIC_OUT, index=False)
categorical_rows.to_csv(CATEGORICAL_OUT, index=False)

print("수치형 변수 영향(표준화 계수, 1표준편차 증가 시 배송일 변화):")
print(numeric_rows[["feature", "coef_per_1sd"]].round(3).to_string(index=False))
print()
print("범주형 변수 영향(기준 범주 대비 배송일 차이, coefficient=0인 행이 기준 범주):")
print(categorical_rows.round(3).to_string(index=False))
print()
print("저장 완료:", NUMERIC_OUT, CATEGORICAL_OUT)
