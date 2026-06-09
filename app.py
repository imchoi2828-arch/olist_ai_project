import os
import pickle
import joblib
import pandas as pd
import gradio as gr

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# 공통 경로
# =========================================================
MODEL_DIR = "model"


# =========================================================
# PART 6. 추천 시스템
# =========================================================
with open(os.path.join(MODEL_DIR, "part6_recommend_model.pkl"), "rb") as f:
    recommend_data = pickle.load(f)

recommend_df = recommend_data["recommend_df"]
similarity_matrix = recommend_data["similarity_matrix"]
product_to_index = recommend_data["product_to_index"]

# 드롭다운에 표시할 상품 목록
# 화면에는 "카테고리 | 상품ID 앞 8자리"만 보여주고,
# 실제 계산에는 전체 product_id를 사용합니다.
product_label_to_id = {}
product_choices = []

for _, row in recommend_df.head(100).iterrows():
    category = row["product_category_name_english"]
    product_id = row["product_id"]

    label = f"{category} | {product_id[:8]}"
    product_choices.append(label)
    product_label_to_id[label] = product_id


def recommend_product(selected_product):
    if selected_product is None or selected_product == "":
        return "상품을 선택해주세요."

    product_id = product_label_to_id.get(selected_product, selected_product)

    if product_id not in product_to_index:
        return "해당 상품을 찾을 수 없습니다."

    idx = product_to_index[product_id]

    # 혹시 같은 product_id가 여러 index로 잡히는 경우 첫 번째 값만 사용
    if hasattr(idx, "iloc"):
        idx = idx.iloc[0]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # 자기 자신 제외 후 상위 5개 추천
    scores = scores[1:6]

    result = ""

    for rank, (i, score) in enumerate(scores, start=1):
        category = recommend_df.iloc[i]["product_category_name_english"]
        recommended_product_id = recommend_df.iloc[i]["product_id"]

        result += (
            f"추천 상품 {rank}\n"
            f"카테고리 : {category}\n"
            f"상품ID : {recommended_product_id[:8]}\n"
            f"유사도 : {score:.2f}\n"
            f"{'-' * 30}\n"
        )

    return result


recommend_interface = gr.Interface(
    fn=recommend_product,
    inputs=gr.Dropdown(
        choices=product_choices,
        label="상품 선택"
    ),
    outputs=gr.Textbox(
        label="추천 결과",
        lines=15
    ),
    title="PART6 추천 시스템",
    description="상품을 선택하면 비슷한 상품을 추천합니다."
)


# =========================================================
# PART 7. 감성 분석
# =========================================================
sentiment_model = load_model(
    os.path.join(MODEL_DIR, "part7_sentiment_model.h5"),
    compile=False
)

with open(os.path.join(MODEL_DIR, "part7_tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)


def predict_sentiment(text):
    if text is None or text.strip() == "":
        return "리뷰를 입력해주세요."

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=100,
        padding="post",
        truncating="post"
    )

    prob = sentiment_model.predict(padded, verbose=0)[0][0]

    if prob >= 0.5:
        return f"긍정 리뷰 ({prob:.2f})"

    return f"부정 리뷰 ({prob:.2f})"


sentiment_interface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="PART7 감성 분석",
    description="""
포르투갈어 리뷰를 입력하면 긍정 / 부정을 예측합니다.

<hr>

<h3>테스트용 리뷰 예시</h3>

<b>긍정 리뷰</b>
<ul>
<li>O produto chegou rápido e em ótimo estado.</li>
<li>Excelente qualidade e ótimo custo-benefício.</li>
<li>Gostei muito do produto e recomendo.</li>
</ul>

<b>부정 리뷰</b>
<ul>
<li>O produto veio danificado.</li>
<li>A entrega demorou muito.</li>
<li>Qualidade abaixo do esperado.</li>
</ul>

※ 위 문장을 복사해서 테스트해보세요.
"""
)


# =========================================================
# PART 8. 매출 예측
# =========================================================
sales_model = joblib.load(
    os.path.join(MODEL_DIR, "part8_sales_model.pkl")
)


def predict_sales(price, freight_value, review_score):
    input_data = pd.DataFrame(
        [[price, freight_value, review_score]],
        columns=["price", "freight_value", "review_score"]
    )

    pred = sales_model.predict(input_data)[0]

    return f"예상 결제 금액 : {round(float(pred), 2)} 헤알"


sales_interface = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(
            label="상품 가격",
            value=0,
            info="100 헤알 ≈ 한국 돈 약 3만원"
        ),
        gr.Number(
            label="배송비",
            value=0,
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
    description="상품 가격, 배송비, 리뷰 점수를 입력하면 예상 결제 금액을 예측합니다."
)


# =========================================================
# PART 9. 고객 이탈 예측
# =========================================================
churn_model = joblib.load(
    os.path.join(MODEL_DIR, "part9_churn_model.pkl")
)

# 원본 화면처럼 입력은 2개만 받습니다.
# 모델이 실제로 요구하는 나머지 feature는 0으로 자동 채웁니다.
feature_names = list(churn_model.feature_names_in_)


def predict_churn(order_count, days_since_last_order):
    input_data = pd.DataFrame(
        [[0] * len(feature_names)],
        columns=feature_names
    )

    if "order_count" in feature_names:
        input_data["order_count"] = order_count

    if "days_since_last_order" in feature_names:
        input_data["days_since_last_order"] = days_since_last_order

    pred = churn_model.predict(input_data)[0]

    if pred == 1:
        return "이탈 가능성 높음"

    return "유지 고객"


churn_interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(
            label="주문 횟수",
            value=0
        ),
        gr.Number(
            label="최근 미구매 기간(일)",
            value=0
        )
    ],
    outputs="text",
    title="PART9 고객 이탈 예측",
    description="""
주문 횟수와 최근 미구매 기간을 입력하면 고객 이탈 가능성을 예측합니다.

※ 모델은 여러 X값으로 학습되었으며, 입력하지 않는 값은 기본값(0)으로 자동 처리됩니다.
"""
)


# =========================================================
# 전체 탭 화면
# =========================================================
# gr.TabbedInterface + gr.Interface 조합을 사용해야
# 원본 화면처럼 Clear / Submit 버튼과 Share via Link 영역이 자동으로 표시됩니다.
demo = gr.TabbedInterface(
    interface_list=[
        recommend_interface,
        sentiment_interface,
        sales_interface,
        churn_interface
    ],
    tab_names=[
        "추천시스템",
        "감성분석",
        "매출 예측",
        "고객 이탈 예측"
    ],
    title="Olist AI Service"
)


if __name__ == "__main__":
    demo.launch()