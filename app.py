# app.py
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import openai

# لو معاك GPT API key
openai.api_key = "YOUR_OPENAI_API_KEY"

st.title("Pixels StudyMate 📘🤖")
st.write("مساعدك الذكي للمذاكرة وفهم ملفات الدرايف")

uploaded_file = st.file_uploader("ارفع ملف PDF", type="pdf")
question = st.text_input("اكتب سؤالك هنا:")

if uploaded_file:
    # استخراج النص من PDF
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # تقسيم النص إلى فقرات صغيرة
    chunks = text.split("\n")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)

    if question:
        # بحث دلالي عن الفقرة الأقرب
        q_embed = model.encode(question, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_embed, embeddings)[0]
        best_idx = scores.argmax()
        context = chunks[best_idx]

        # توليد إجابة باستخدام GPT (اختياري)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أنت مساعد دراسي ذكي لطلبة هندسة."},
                {"role": "user", "content": f"السؤال: {question}\n\nالمصدر: {context}"}
            ]
        )
        answer = response['choices'][0]['message']['content']
        st.markdown("### 🤖 إجابة المساعد:")
        st.write(answer)
