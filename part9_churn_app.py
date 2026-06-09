import gradio as gr
import joblib
import pandas as pd

# 모델 불러오기
model = joblib.load(
    "model/part9_churn_model.pkl"
)

# 모델 X값 이름 가져오기
feature_names = model.feature_names_in_

# 고객 이탈 예측 함수
def predict_churn(order_count, days):

    # 입력 데이터 기본값 생성
    input_data = pd.DataFrame(
        [[0] * len(feature_names)],
        columns=feature_names
    )

    # 입력값 넣기
    if "order_count" in feature_names:
        input_data["order_count"] = order_count

    if "days_since_last_order" in feature_names:
        input_data["days_since_last_order"] = days

    # AI 예측
    pred = model.predict(input_data)[0]

    if pred == 1:
        return "이탈 가능성 높음"

    return "유지 고객"

# Gradio 화면 생성
demo = gr.Interface(

    fn=predict_churn,

    inputs=[

        gr.Number(
            label="주문 횟수"
        ),

        gr.Number(
            label="최근 미구매 기간(일)"
        )
    ],

    outputs="text",

    title="PART9 고객 이탈 예측",

    description="""
주문 횟수와 최근 미구매 기간을 입력하면
고객 이탈 가능성을 예측합니다.

※ 모델은 여러 X값으로 학습되었으며,
입력하지 않는 값은 기본값(0)으로 자동 처리합니다.
"""
)

# 실행
demo.launch()