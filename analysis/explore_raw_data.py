# requires: pandas
import pandas as pd

RAW_PATH = "data/raw/olist_delivery_orders_sample.csv"

df = pd.read_csv(RAW_PATH)

print("shape:", df.shape)
print()

print("dtypes:")
print(df.dtypes)
print()

n_dup_id = df["order_id"].duplicated().sum()
n_unique_id = df["order_id"].nunique()
print(f"order_id 총 행 수: {len(df)}, 고유값 수: {n_unique_id}, 중복 행 수: {n_dup_id}")
print()

summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_missing": df.isna().sum(),
    "missing_rate": (df.isna().mean() * 100).round(2),
    "n_unique": df.nunique(),
})
print("변수 요약:")
print(summary)
print()

print("수치형 변수 describe:")
print(df.describe().T)
print()

print("범주형 변수 상위 값:")
for col in df.select_dtypes(include="object").columns:
    if col == "order_id":
        continue
    print(f"- {col}: {df[col].value_counts().head(5).to_dict()}")
