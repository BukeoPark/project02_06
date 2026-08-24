# requires: pandas, numpy, scikit-learn, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
FIG_DIR = "outputs/figures"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

distance_median = train_df["distance_km"].median()
n_missing_train = train_df["distance_km"].isna().sum()
n_missing_test = test_df["distance_km"].isna().sum()

train_df = train_df.copy()
test_df = test_df.copy()
train_df["distance_km"] = train_df["distance_km"].fillna(distance_median)
test_df["distance_km"] = test_df["distance_km"].fillna(distance_median)

print(f"distance_km 결측치 대체 기준(학습 데이터 중앙값): {distance_median:.3f}")
print(f"train 결측 대체 건수: {n_missing_train}, test 결측 대체 건수: {n_missing_test}")

X_train = train_df[["distance_km"]]
y_train = train_df["delivery_days"]
X_test = test_df[["distance_km"]]
y_test = test_df["delivery_days"]

model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_
print(f"회귀식: delivery_days = {intercept:.4f} + {slope:.6f} * distance_km")
print(f"distance_km 100km 증가 시 예측 배송기간 변화: {slope * 100:.3f}일")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R^2: {r2:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_test["distance_km"], y_test, s=8, alpha=0.3, color="#55A868", label="test data")
x_line = np.linspace(X_test["distance_km"].min(), X_test["distance_km"].max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
ax.plot(x_line, y_line, color="#C44E52", linewidth=2, label="regression line")
ax.set_xlabel("distance_km")
ax.set_ylabel("delivery_days")
ax.set_title("distance_km vs delivery_days (test set) with regression line")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/regression_distance_km_vs_delivery_days.png", dpi=150)
plt.close(fig)
