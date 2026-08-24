# requires: streamlit, pandas, numpy, joblib, scikit-learn
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "models/final_model.joblib"
META_PATH = "models/final_model_meta.json"

st.set_page_config(page_title="배송 예상일 조회", page_icon="📦")


@st.cache_resource
def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_model_and_meta()

st.title("📦 주문 배송 예상일 조회")
st.caption(
    f"다중선형회귀 모델 (distance_km 로그 전환 적용) · 테스트 MAE {meta['test_mae']:.2f}일, "
    f"RMSE {meta['test_rmse']:.2f}일, R² {meta['test_r2']:.2f}"
)

st.subheader("주문 조건 입력")

col1, col2 = st.columns(2)
with col1:
    customer_state = st.selectbox("고객 지역(주)", meta["customer_states"])
    seller_state = st.selectbox("판매자 지역(주)", meta["seller_states"])
    primary_category = st.selectbox("상품 카테고리", meta["primary_categories"])
    purchase_weekday = st.selectbox("구매 요일", meta["purchase_weekdays"])
    purchase_hour = st.slider("구매 시각(시)", 0, 23, 14)

with col2:
    distance_km = st.number_input("배송 거리(km)", min_value=0.0, value=400.0, step=10.0)
    item_count = st.number_input("상품 개수", min_value=1, value=1, step=1)
    seller_count = st.number_input("판매자 수", min_value=1, value=1, step=1)
    total_price_brl = st.number_input("총 상품 금액(BRL)", min_value=0.0, value=100.0, step=1.0)
    total_freight_brl = st.number_input("총 배송비(BRL)", min_value=0.0, value=20.0, step=1.0)
    total_weight_kg = st.number_input("총 무게(kg)", min_value=0.0, value=1.0, step=0.1)
    total_volume_l = st.number_input("총 부피(L)", min_value=0.0, value=5.0, step=0.5)

if st.button("예상 배송일 계산", type="primary"):
    input_row = pd.DataFrame([{
        "distance_km_log": np.log1p(distance_km),
        "item_count": item_count,
        "seller_count": seller_count,
        "total_price_brl": total_price_brl,
        "total_freight_brl": total_freight_brl,
        "total_weight_kg": total_weight_kg,
        "total_volume_l": total_volume_l,
        "purchase_hour": purchase_hour,
        "customer_state": customer_state,
        "seller_state": seller_state,
        "primary_category": primary_category,
        "purchase_weekday": purchase_weekday,
    }])

    predicted_days = float(model.predict(input_row)[0])
    lower_bound = max(0.0, predicted_days + meta["residual_q10"])
    upper_bound = max(0.0, predicted_days + meta["residual_q90"])

    st.subheader("예측 결과")
    st.metric("예상 배송일", f"{predicted_days:.1f}일")
    st.write(
        f"실제로는 **약 {lower_bound:.1f}일 ~ {upper_bound:.1f}일** 사이에 도착할 가능성이 큽니다 "
        f"(과거 테스트 데이터 오차 분포의 10~90% 구간 기준)."
    )

    delay_days = upper_bound - predicted_days
    early_days = predicted_days - lower_bound
    st.write(
        f"- 예상보다 **늦어질 경우** 최대 약 {delay_days:.1f}일 더 걸릴 수 있습니다.\n"
        f"- 예상보다 **빨리 도착할 경우** 최대 약 {early_days:.1f}일 당겨질 수 있습니다."
    )

    st.caption("이 예측은 학습에 사용한 데이터 범위를 벗어난 조건에서는 정확도가 낮아질 수 있습니다.")
