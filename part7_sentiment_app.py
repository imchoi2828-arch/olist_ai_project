# part7_sentiment_app.py

import gradio as gr
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 모델 불러오기
model = load_model(
    "model/part7_sentiment_model.h5"
)

# tokenizer 불러오기
with open(
    "model/part7_tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(f)

# 감성 분석 함수
def predict_sentiment(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=100,
        padding="post"
    )

    pred = model.predict(padded)[0][0]

    if pred >= 0.5:
        return f"긍정 리뷰 ({pred:.2f})"

    return f"부정 리뷰 ({pred:.2f})"

# Gradio 화면
demo = gr.Interface(
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

# 실행
demo.launch()




# 테스트용 리뷰 예시

# | 구분 | 포르투갈어 리뷰                                 | 해석 |
# |------|------------------------------------------------|------|
# | 긍정 | O produto chegou rápido e em ótimo estado.    | 상품이 빠르고 좋은 상태로 도착했다 |
# | 긍정 | Excelente qualidade e ótimo custo-benefício.  | 품질이 매우 좋고 가성비가 훌륭하다 |
# | 긍정 | Gostei muito do produto e recomendo.          | 상품이 매우 마음에 들며 추천한다 |
# | 부정 | O produto veio danificado.                    | 상품이 파손되어 왔다 |
# | 부정 | A entrega demorou muito.                      | 배송이 너무 오래 걸렸다 |
# | 부정 | Qualidade abaixo do esperado.                 | 품질이 기대 이하이다 |