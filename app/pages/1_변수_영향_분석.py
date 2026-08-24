# requires: streamlit, pandas, matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from font_utils import apply_korean_font

NUMERIC_PATH = "outputs/tables/feature_importance_numeric.csv"
CATEGORICAL_PATH = "outputs/tables/feature_importance_categorical.csv"

st.set_page_config(page_title="변수 영향 분석", page_icon="📊")
apply_korean_font()

st.title("📊 변수별 영향 분석")
st.caption(
    "최종 모델(distance_km 로그 전환, 12개 변수)과 동일한 구성으로, "
    "수치형은 표준화(StandardScaler), 범주형은 기준 범주를 뺀 원-핫 인코딩을 적용해 "
    "계수를 해석 가능한 형태로 다시 학습한 결과입니다. (학습 데이터 기준)"
)


@st.cache_data
def load_tables():
    numeric_df = pd.read_csv(NUMERIC_PATH)
    categorical_df = pd.read_csv(CATEGORICAL_PATH)
    return numeric_df, categorical_df


numeric_df, categorical_df = load_tables()

st.subheader("수치형 변수 영향")
st.write("각 변수가 **1표준편차만큼 증가**할 때, 다른 변수는 그대로 두었을 때 예상 배송일이 며칠 변하는지를 나타냅니다.")

numeric_sorted = numeric_df.sort_values("coef_per_1sd")
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#C44E52" if v > 0 else "#4C72B0" for v in numeric_sorted["coef_per_1sd"]]
ax.barh(numeric_sorted["feature"], numeric_sorted["coef_per_1sd"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("delivery_days 변화량 (1표준편차 증가 시, 일)")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "빨간 막대는 값이 커질수록 배송이 오래 걸리는 방향, 파란 막대는 값이 커질수록 배송이 빨라지는 방향입니다. "
    "distance_km_log(거리)가 가장 큰 영향을 주고, seller_count·item_count는 값이 클수록 오히려 배송이 "
    "빨라지는 경향(묶음 배송 처리 효과 등으로 추정)입니다."
)

st.divider()

st.subheader("범주형 변수 영향")
st.write("선택한 변수의 각 범주가, 기준 범주(coefficient=0) 대비 예상 배송일을 며칠 더/덜 걸리게 하는지 보여줍니다.")

variable_options = categorical_df["variable"].unique().tolist()
selected_var = st.selectbox("변수 선택", variable_options)

var_df = categorical_df[categorical_df["variable"] == selected_var].copy()
reference_category = var_df.loc[var_df["is_reference"], "category"].iloc[0]

top_n = st.slider("표시할 범주 수 (영향 큰 순)", 5, min(40, len(var_df)), min(20, len(var_df)))
var_df["abs_coef"] = var_df["coefficient"].abs()
show_df = var_df.sort_values("abs_coef", ascending=False).head(top_n).sort_values("coefficient")

fig2, ax2 = plt.subplots(figsize=(7, max(3, 0.28 * len(show_df))))
colors2 = ["#C44E52" if v > 0 else ("#999999" if v == 0 else "#4C72B0") for v in show_df["coefficient"]]
ax2.barh(show_df["category"], show_df["coefficient"], color=colors2)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel(f"delivery_days 변화량 (기준 범주 '{reference_category}' 대비, 일)")
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.caption(f"기준 범주: **{reference_category}** (계수 0) · 전체 {len(var_df)}개 범주 중 영향 큰 {len(show_df)}개만 표시")

with st.expander("전체 범주 계수 표 보기"):
    st.dataframe(var_df[["category", "coefficient", "is_reference"]].sort_values("coefficient", ascending=False),
                 use_container_width=True, hide_index=True)
