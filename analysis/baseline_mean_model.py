# requires: pandas, numpy, scikit-learn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

baseline_pred = train_df["delivery_days"].mean()
y_true = test_df["delivery_days"]
y_pred = np.full(len(y_true), baseline_pred)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"기준 모델 예측값(train 평균 delivery_days): {baseline_pred:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R^2: {r2:.3f}")
