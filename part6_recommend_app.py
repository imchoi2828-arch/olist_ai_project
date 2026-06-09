# part6_recommend_app.py

import gradio as gr
import pickle



# 저장된 추천 모델 불러오기

with open(
    "model/part6_recommend_model.pkl",
    "rb"
) as f:

    model_data = pickle.load(f)




# 저장된 데이터 가져오기

recommend_df = model_data["recommend_df"]

similarity_matrix = model_data["similarity_matrix"]

product_to_index = model_data["product_to_index"]




# 드롭다운 상품 목록 생성
product_options = []

for _, row in recommend_df.head(100).iterrows():

    # 상품 카테고리
    english_category = row[
        "product_category_name_english"
    ]

    # 상품 ID
    product_id = row["product_id"]

    # 화면에 보여줄 이름
    label = (
        f"{english_category} | "
        f"{product_id[:8]}"
    )

    # 실제 전달 값
    value = product_id

    # 목록 저장
    product_options.append(
        (label, value)
    )



# 추천 함수
def recommend_product(product_id):

    # 선택 상품 index 찾기
    idx = product_to_index[product_id]

    # 유사도 가져오기
    similarity_scores = list(
        enumerate(similarity_matrix[idx])
    )

    # 유사도 높은 순 정렬
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # 자기 자신 제외 후 상위 5개
    similarity_scores = similarity_scores[1:6]

    result = ""

    # 추천 결과 생성
    for rank, (i, score) in enumerate(
        similarity_scores,
        start=1
    ):

        # 추천 상품 카테고리
        category = recommend_df.iloc[i][
            "product_category_name_english"
        ]

        # 추천 상품 ID
        recommended_product = recommend_df.iloc[i][
            "product_id"
        ]

        # 결과 문자열 생성
        result += (
            f"추천 상품 {rank}\n"
            f"카테고리 : {category}\n"
            f"상품ID : {recommended_product[:8]}\n"
            f"유사도 : {score:.2f}\n"
            f"{'-'*30}\n"
        )

    return result

# Gradio 화면 생성

demo = gr.Interface(
    fn=recommend_product,
    inputs=gr.Dropdown(
        choices=product_options,
        label="상품 선택"
    ),

    outputs=gr.Textbox(
        label="추천 결과",
        lines=15
    ),

    title="PART6 추천 시스템",
    
    description="""
    상품을 선택하면
    비슷한 상품을 추천합니다.
    """
)

# 실행
demo.launch()