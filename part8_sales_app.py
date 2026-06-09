# part8_sales_app.py

import gradio as gr
import joblib
import numpy as np

# 저장된 매출 예측 모델 불러오기
model = joblib.load(
    "model/part8_sales_model.pkl"
)

# 매출 예측 함수
def predict_sales(price, freight, review_score):

    # X값 3개만 사용
    data = np.array([[
        price,
        freight,
        review_score
    ]])

    pred = model.predict(data)

    return f"예상 결제 금액 : {round(float(pred[0]), 2)} 헤알"

# Gradio 화면 생성
demo = gr.Interface(

    fn=predict_sales,

    inputs=[

        gr.Number(
            label="상품 가격",
            info="100 헤알 ≈ 한국 돈 약 3만원"
        ),

        gr.Number(
            label="배송비",
            info="20 헤알 ≈ 한국 돈 약 6천원"
        ),

        gr.Slider(
            minimum=1,
            maximum=5,
            value=5,
            step=1,
            label="리뷰 점수"
        )

    ],

    outputs=gr.Textbox(
        label="예측 결과"
    ),

    title="PART8 매출 예측",

    description="""
상품 가격, 배송비, 리뷰 점수를 입력하면 예상 결제 금액을 예측합니다.
"""
)

# 실행
demo.launch()