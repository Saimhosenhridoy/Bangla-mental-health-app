import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from explain import explain_text_interactive
from model import DISPLAY_LABELS, LABEL_DESCRIPTIONS
from model_loader import predict_text
from settings import settings
from text_utils import clean_text, contains_crisis_language

st.set_page_config(
    page_title="Bangla Mental Health Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_MESSAGES = {
    "Depressive": "This text shows linguistic signs linked to distress. Support from someone you trust can help.",
    "Non_depressive": "This text does not show clear depressive markers. Keep balanced routines and self-care.",
    "Positive": "This text reflects a positive mental state. Keep nurturing that outlook.",
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Hind+Siliguri:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Hind Siliguri', sans-serif !important; }
div[data-testid="stSidebar"], footer, div[data-testid="stFooter"] { display: none !important; }
.stApp { background: #B9C7D2; }
.block-container { max-width: 1000px; padding-top: 0.8rem; padding-bottom: 1.5rem; }

.hero {
  padding: 1.4rem 1.2rem; border-radius: 18px; background: #2B3034; color: #fff;
  border: 4px solid #627C8C; text-align: center; margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 1.7rem; }
.hero p { color: #B9C7D2; margin: 0.45rem auto 0; max-width: 560px; font-size: 0.9rem; }

.section-header {
  font-size: 1.15rem; font-weight: 700; color: #2B3034;
  margin: 1.1rem 0 0.55rem; padding-bottom: 0.3rem; border-bottom: 4px solid #E93C35;
}

.result-card, .chart-container {
  height: 220px; min-height: 220px; max-height: 220px;
  border-radius: 16px; padding: 1rem 1.1rem;
  box-sizing: border-box; overflow: hidden;
}
.result-card {
  background: #627C8C; color: #fff; border: 3px solid #989398;
  display: flex; flex-direction: column; justify-content: center;
}
.result-card h2 { margin: 0.2rem 0; font-size: 1.35rem; }
.result-card .eyebrow { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
.result-card .confidence { font-size: 1rem; font-weight: 700; margin-top: 0.3rem; }
.result-card .description { color: #B9C7D2; font-size: 0.84rem; margin-top: 0.4rem; }
.class-msg {
  margin-top: 0.55rem; padding: 0.55rem 0.7rem; border-radius: 10px;
  background: #2B3034; color: #B9C7D2; font-size: 0.82rem;
}
.chart-container {
  background: #989398; border: 3px solid #627C8C;
}
.chart-container [data-testid="stVerticalBlock"] { gap: 0 !important; }
.chart-container [data-testid="stMarkdown"] { display: none !important; }

.example-container {
  background: #627C8C; border: 3px solid #989398;
  border-radius: 14px; padding: 0.75rem; margin-bottom: 0.9rem;
}
div[data-testid="stTextArea"] textarea {
  background: #ffffff; color: #000000; border: 3px solid #627C8C;
  border-radius: 14px; min-height: 140px;
  font-family: 'Hind Siliguri', sans-serif !important;
}
div.stButton > button {
  border-radius: 12px; font-weight: 700; min-height: 44px;
  background: #2B3034; color: #B9C7D2; border: 0;
}
.example-container div.stButton > button,
.example-container div.stButton > button p,
.example-container div.stButton > button span {
  background: #B9C7D2 !important;
  color: #000000 !important;
  border: 2px solid #2B3034 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div style="font-size:2rem;margin-bottom:0.25rem;">🧠</div>
  <h1>Bangla Mental Health Classifier</h1>
  <p>AI-powered Bengali text analysis for mental health awareness. Not a clinical diagnosis.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="example-container">', unsafe_allow_html=True)
st.markdown(
    '<p class="section-header" style="margin-top:0;font-size:1rem;color:#B9C7D2;">Try an example</p>',
    unsafe_allow_html=True,
)


def use_example(example_text):
    st.session_state["mental_text"] = example_text


examples = [
    ("Depressive", "কয়েকদিন ধরে আমার কোনো কাজে মন বসছে না।"),
    ("Non_depressive", "আজ সারাদিন কাজ করেছি, এখন বিশ্রাম নিচ্ছি।"),
    ("Positive", "আজ আমি খুব আনন্দিত, বন্ধুদের সঙ্গে সময় কাটিয়েছি।"),
]
cols = st.columns(3)
for col, (label, text) in zip(cols, examples):
    with col:
        st.button(
            label,
            use_container_width=True,
            on_click=use_example,
            args=(text,),
            key=f"ex_{label}",
            type="secondary",
        )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<p class="section-header">Enter your text</p>', unsafe_allow_html=True)
user_text = st.text_area(
    "Enter your text",
    placeholder="বাংলায় আপনার ভাবনা লিখুন...",
    label_visibility="collapsed",
    key="mental_text",
    max_chars=settings.max_text_chars,
)
analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)

if analyze_button:
    cleaned_text = clean_text(user_text)
    if len(cleaned_text) < 5:
        st.error("Please enter at least one meaningful Bengali sentence.")
        st.stop()
    if contains_crisis_language(cleaned_text):
        st.error(
            "This text may contain indicators of self-harm or suicidal thoughts. "
            "If you are in immediate distress, contact a trusted person, emergency services, "
            "or a qualified professional right away."
        )
    try:
        with st.spinner("Analyzing text..."):
            result = predict_text(cleaned_text)
    except Exception as error:
        st.error(f"Model could not be loaded: {error}")
        st.stop()

    predicted_label = result["pred_label"]
    display_label = DISPLAY_LABELS[predicted_label]
    confidence = result["confidence"] * 100
    short_msg = CLASS_MESSAGES.get(predicted_label, "")

    st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)
    result_column, chart_column = st.columns(2, gap="large")

    with result_column:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="eyebrow">Prediction</div>
              <h2>{display_label}</h2>
              <div class="confidence">Confidence: {confidence:.2f}%</div>
              <p class="description">{LABEL_DESCRIPTIONS[predicted_label]}</p>
              <div class="class-msg">{short_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with chart_column:
        probability_rows = [
            {"Class": DISPLAY_LABELS[label], "Probability (%)": round(prob * 100, 2)}
            for label, prob in result["probabilities"].items()
        ]
        probability_df = pd.DataFrame(probability_rows).set_index("Class")
        st.bar_chart(probability_df, color="#2B3034", height=220)

    st.markdown('<p class="section-header">SHAP Explanation</p>', unsafe_allow_html=True)
    st.caption("Word-level contributions for Depressive, Non_depressive, and Positive")
    try:
        with st.spinner("Generating SHAP explanation..."):
            shap_html = explain_text_interactive(cleaned_text, result["pred_index"])
            components.html(shap_html, height=700, scrolling=True)
    except Exception as error:
        st.warning(f"Prediction succeeded, but SHAP could not be generated: {error}")

    st.info(
        "This is a research model. It is not for diagnosis or as a substitute for professional care."
    )