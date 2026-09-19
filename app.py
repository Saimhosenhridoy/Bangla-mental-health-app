import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from explain import explain_text_interactive
from model import DISPLAY_LABELS, LABEL_DESCRIPTIONS
from model_loader import get_device, predict_text
from settings import settings
from text_utils import clean_text, contains_crisis_language

st.set_page_config(page_title="Bangla Mental Health Classifier", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Hind+Siliguri:wght@400;500;600;700&display=swap');
* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#root > div:nth-child(1) > div > div > div > div > section > div, button[kid="collapse-button"], div[data-testid="stSidebar"], footer, div[data-testid="stFooter"] { display: none !important; }
.stApp { background: linear-gradient(135deg, #e6f0f2 0%, #d7e5d2 50%, #a8d06f 100%); min-height: 100vh; }
.block-container { max-width: 1000px; padding-top: 1rem; padding-bottom: 2rem; }
.hero { padding: 0.9rem; border-radius: 14px; background: linear-gradient(135deg, #a8d06f 0%, #94a8af 50%, #a8d06f 100%); color: white; box-shadow: 0 8px 25px rgba(168, 208, 111, 0.3); margin-bottom: 0.9rem; text-align: center; border: 2px solid #94a8af; }
.hero h1 { margin: 0; font-size: 1.4rem; font-weight: 800; }
.hero p { color: #e6f0f2; font-size: 0.8rem; max-width: 500px; margin: 0.35rem auto 0; }
.result-card { border-radius: 14px; padding: 1.2rem; color: white; background: linear-gradient(135deg, #94a8af 0%, #a8d06f 100%); box-shadow: 0 10px 30px rgba(148, 168, 175, 0.3); border: 2px solid #a8d06f; min-height: 180px; display: flex; flex-direction: column; justify-content: center; }
.result-card h2 { margin: 0.25rem 0; font-size: 1.4rem; font-weight: 800; }
.result-card .eyebrow { font-size: 0.65rem; opacity: 0.9; font-weight: 600; text-transform: uppercase; }
.result-card .confidence { font-size: 0.95rem; font-weight: 700; margin-top: 0.35rem; }
.result-card .description { color: #e6f0f2; font-size: 0.8rem; margin-top: 0.5rem; }
.shap-wrap { display: flex; flex-wrap: wrap; gap: 8px; line-height: 2.2; padding: 10px 0 14px; }
.shap-token { display: inline-block; padding: 3px 9px; border: 1px solid; border-radius: 8px; color: #1e293b; font-weight: 600; cursor: help; }
.legend { color: #64748b; font-size: 0.85rem; margin-top: 6px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 5px 0 12px; vertical-align: middle; }
div[data-testid="stTextArea"] textarea { background: rgba(255, 255, 255, 0.95); border-radius: 12px; border: 2px solid #94a8af; min-height: 130px; font-size: 0.9rem; font-family: 'Hind Siliguri', sans-serif !important; }
div.stButton > button { border: 0; border-radius: 11px; font-weight: 700; min-height: 42px; background: linear-gradient(135deg, #a8d06f 0%, #94a8af 100%); box-shadow: 0 5px 16px rgba(168, 208, 111, 0.3); }
.section-header { font-size: 1.05rem; font-weight: 700; color: #94a8af; margin: 0.9rem 0 0.45rem; padding-bottom: 0.25rem; border-bottom: 2px solid #a8d06f; }
.chart-container { background: white; border-radius: 14px; padding: 1.2rem; box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06); border: 2px solid #94a8af; min-height: 180px; display: flex; flex-direction: column; justify-content: center; }
.example-btn { border-radius: 10px; font-weight: 600; min-height: 36px; font-size: 0.85rem; background: linear-gradient(135deg, #d7e5d2 0%, #e6f0f2 100%); border: 2px solid #94a8af; color: #94a8af; }
@media (max-width: 768px) { .stColumns > div:first-child, .stColumns > div:last-child { width: 100% !important; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div style="font-size:1.6rem; margin-bottom:0.25rem">🧠</div>
    <h1>Bangla Mental Health Classifier</h1>
    <p>AI-powered Bengali text analysis for mental health awareness.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="section-header">Try an example</p>', unsafe_allow_html=True)

def use_example(example_text):
    st.session_state["mental_text"] = example_text

examples = [("Depressive", "কয়েকদিন ধরে আমার কোনো কাজে মন বসছে না, সবকিছু খুব অর্থহীন মনে হচ্ছে।"), ("Normal", "আজ সারাদিন কাজ করেছি, এখন বাসায় ফিরে বিশ্রাম নিচ্ছি।"), ("Positive", "আজ আমি খুব আনন্দিত, অনেকদিন পর বন্ধুদের সঙ্গে সুন্দর সময় কাটিয়েছি।")]

example_columns = st.columns(3)
for column, example in zip(example_columns, examples):
    button_label, example_text = example
    with column:
        st.button(button_label, use_container_width=True, on_click=use_example, args=(example_text,), key=f"ex_{button_label}", type="secondary")

st.markdown('<p class="section-header">Enter your text</p>', unsafe_allow_html=True)
user_text = st.text_area("Enter your text", placeholder="Write your thoughts in Bengali here...", label_visibility="collapsed", key="mental_text", max_chars=settings.max_text_chars)

analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)

if analyze_button:
    cleaned_text = clean_text(user_text)
    if len(cleaned_text) < 5:
        st.error("Please enter at least one meaningful Bengali sentence.")
        st.stop()
    if contains_crisis_language(cleaned_text):
        st.error("This text may contain indicators of self-harm or suicidal thoughts. If you are in immediate distress, please reach out to a trusted person, emergency services, or a qualified mental health professional right away.")
    try:
        with st.spinner("Analyzing text..."):
            result = predict_text(cleaned_text)
    except Exception as error:
        st.error(f"Model could not be loaded: {error}")
        st.stop()

    predicted_label = result["pred_label"]
    display_label = DISPLAY_LABELS[predicted_label]
    confidence = result["confidence"] * 100

    st.markdown("---")
    st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)

    result_column, chart_column = st.columns([1.4, 1], gap="large")

    with result_column:
        st.markdown(f"""
        <div class="result-card">
            <div class="eyebrow">Prediction</div>
            <h2>{display_label}</h2>
            <div class="confidence">Confidence: {confidence:.2f}%</div>
            <p class="description">{LABEL_DESCRIPTIONS[predicted_label]}</p>
        </div>
        """, unsafe_allow_html=True)

    with chart_column:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Class Probabilities")
        probability_rows = [{"Class": DISPLAY_LABELS[label], "Probability (%)": round(probability * 100, 2)} for label, probability in result["probabilities"].items()]
        probability_df = pd.DataFrame(probability_rows).set_index("Class")
        st.bar_chart(probability_df, color="#94a8af")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">SHAP Explanation</p>', unsafe_allow_html=True)
    st.caption(f"Word-level contributions for the predicted class: '{display_label}'")

    try:
        with st.spinner("Generating SHAP explanation..."):
            shap_html = explain_text_interactive(cleaned_text, result["pred_index"])
            components.html(shap_html, height=480, scrolling=True)
    except Exception as error:
        st.warning(f"Prediction succeeded, but SHAP could not be generated: {error}")

    st.markdown("---")
    st.info("**Important:** This is a research model. It is not intended for diagnosis, risk assessment, or as a substitute for professional mental health advice. If you are experiencing persistent or severe distress, please consult a qualified mental health professional.")