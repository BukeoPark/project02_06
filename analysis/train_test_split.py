# requires: pandas, scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/olist_delivery_orders_sample.csv"
PROCESSED_DIR = "data/processed"

df = pd.read_csv(RAW_PATH)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

print(f"train 행 수: {len(train_df)}")
print(f"test 행 수: {len(test_df)}")
print(f"train delivery_days 평균: {train_df['delivery_days'].mean():.3f}")
print(f"test delivery_days 평균: {test_df['delivery_days'].mean():.3f}")
