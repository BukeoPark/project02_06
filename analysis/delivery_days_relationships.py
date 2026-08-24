# requires: pandas, matplotlib
import pandas as pd
import matplotlib.pyplot as plt

RAW_PATH = "data/raw/olist_delivery_orders_sample.csv"
FIG_DIR = "outputs/figures"

df = pd.read_csv(RAW_PATH)

mean_val = df["delivery_days"].mean()
min_val = df["delivery_days"].min()
max_val = df["delivery_days"].max()
print(f"delivery_days 평균: {mean_val:.3f}")
print(f"delivery_days 최소값: {min_val:.3f}")
print(f"delivery_days 최대값: {max_val:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df["delivery_days"].dropna(), bins=40, color="#4C72B0", edgecolor="white")
ax.set_xlabel("delivery_days")
ax.set_ylabel("count")
ax.set_title("Histogram of delivery_days")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/delivery_days_hist.png", dpi=150)
plt.close(fig)

scatter_targets = ["distance_km", "total_weight_kg", "total_freight_brl"]
for col in scatter_targets:
    sub = df[[col, "delivery_days"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub[col], sub["delivery_days"], s=8, alpha=0.3, color="#55A868")
    ax.set_xlabel(col)
    ax.set_ylabel("delivery_days")
    ax.set_title(f"{col} vs delivery_days")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/scatter_{col}_vs_delivery_days.png", dpi=150)
    plt.close(fig)

    corr = sub[col].corr(sub["delivery_days"])
    print(f"{col} vs delivery_days 상관계수: {corr:.3f}")

print("figures saved to", FIG_DIR)
