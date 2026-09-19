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
.stApp { background: #B9C7D2; min-height: 100vh; }
.block-container { max-width: 1000px; padding-top: 0.8rem; padding-bottom: 1.5rem; }
.hero { padding: 1.5rem; border-radius: 20px; background: #2B3034; color: white; box-shadow: 0 15px 50px rgba(43, 48, 52, 0.5); margin-bottom: 1.2rem; text-align: center; border: 4px solid #627C8C; position: relative; overflow: hidden; }
.hero::before { content: "🧠"; position: absolute; top: -30px; left: -30px; font-size: 10rem; opacity: 0.08; }
.hero::after { content: "💚"; position: absolute; bottom: -20px; right: -20px; font-size: 8rem; opacity: 0.08; }
.hero h1 { margin: 0; font-size: 1.8rem; font-weight: 800; position: relative; z-index: 1; }
.hero p { color: #B9C7D2; font-size: 0.9rem; max-width: 600px; margin: 0.5rem auto 0; position: relative; z-index: 1; }
.result-card { border-radius: 18px; padding: 1.5rem; color: white; background: #627C8C; box-shadow: 0 12px 40px rgba(98, 124, 140, 0.4); border: 3px solid #989398; min-height: 180px; display: flex; flex-direction: column; justify-content: center; }
.result-card h2 { margin: 0.3rem 0; font-size: 1.6rem; font-weight: 800; }
.result-card .eyebrow { font-size: 0.75rem; opacity: 0.9; font-weight: 600; text-transform: uppercase; }
.result-card .confidence { font-size: 1.1rem; font-weight: 700; margin-top: 0.5rem; }
.result-card .description { color: #B9C7D2; font-size: 0.9rem; margin-top: 0.7rem; }
.shap-wrap { display: flex; flex-wrap: wrap; gap: 8px; line-height: 2.2; padding: 10px 0 14px; }
.shap-token { display: inline-block; padding: 3px 9px; border: 1px solid; border-radius: 8px; color: #1e293b; font-weight: 600; cursor: help; }
.legend { color: #64748b; font-size: 0.85rem; margin-top: 6px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 5px 0 12px; vertical-align: middle; }
div[data-testid="stTextArea"] textarea { background: #B9C7D2; border-radius: 14px; border: 3px solid #627C8C; min-height: 150px; font-size: 1rem; font-family: 'Hind Siliguri', sans-serif !important; color: #2B3034; }
div.stButton > button { border: 0; border-radius: 14px; font-weight: 700; min-height: 48px; background: #2B3034; box-shadow: 0 8px 25px rgba(43, 48, 52, 0.4); color: #B9C7D2; }
.section-header { font-size: 1.2rem; font-weight: 700; color: #2B3034; margin: 1.2rem 0 0.6rem; padding-bottom: 0.4rem; border-bottom: 4px solid #E93C35; }
.chart-container { background: #989398; border-radius: 18px; padding: 1.5rem; box-shadow: 0 6px 25px rgba(152, 147, 152, 0.3); border: 3px solid #627C8C; min-height: 180px; display: flex; flex-direction: column; justify-content: center; }
.example-container { background: #627C8C; border-radius: 14px; padding: 0.8rem; margin-bottom: 1rem; border: 3px solid #989398; }
.example-btn { border-radius: 10px; font-weight: 600; min-height: 38px; font-size: 0.85rem; background: #B9C7D2; border: 2px solid #2B3034; color: #2B3034; }
.depressive-msg { background: linear-gradient(135deg, #E93C35 0%, #989398 100%); color: white; padding: 15px; border-radius: 12px; margin-top: 10px; font-size: 0.9rem; }
.normal-msg { background: linear-gradient(135deg, #627C8C 0%, #989398 100%); color: white; padding: 15px; border-radius: 12px; margin-top: 10px; font-size: 0.9rem; }
.positive-msg { background: linear-gradient(135deg, #627C8C 0%, #B9C7D2 100%); color: white; padding: 15px; border-radius: 12px; margin-top: 10px; font-size: 0.9rem; }
@media (max-width: 768px) { .stColumns > div:first-child, .stColumns > div:last-child { width: 100% !important; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div style="font-size:2.5rem; margin-bottom:0.4rem">🧠 💚 🤝</div>
    <h1>Bangla Mental Health Classifier</h1>
    <p>AI-powered Bengali text analysis for mental health awareness and early detection.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="example-container">', unsafe_allow_html=True)
st.markdown('<p class="section-header" style="margin-top:0; font-size:1rem; color:#B9C7D2;">💡 Try an example</p>', unsafe_allow_html=True)

def use_example(example_text):
    st.session_state["mental_text"] = example_text

examples = [("Depressive", "কয়েকদিন ধরে আমার কোনো কাজে মন বসছে না।"), ("Normal", "আজ সারাদিন কাজ করেছি, এখন বিশ্রাম নিচ্ছি।"), ("Positive", "আজ আমি খুব আনন্দিত, বন্ধুদের সঙ্গে সময় কাটিয়েছি।")]

example_columns = st.columns(3)
for column, example in zip(example_columns, examples):
    button_label, example_text = example
    with column:
        st.button(button_label, use_container_width=True, on_click=use_example, args=(example_text,), key=f"ex_{button_label}", type="secondary")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="section-header">✍️ Enter your text</p>', unsafe_allow_html=True)
user_text = st.text_area("Enter your text", placeholder="Write your thoughts in Bengali here...", label_visibility="collapsed", key="mental_text", max_chars=settings.max_text_chars)

analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

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
    st.markdown('<p class="section-header">📊 Prediction Result</p>', unsafe_allow_html=True)

    result_column, chart_column = st.columns([1.4, 1], gap="large")

    with result_column:
        st.markdown(f"""
        <div class="result-card">
            <div class="eyebrow">Prediction</div>
            <h2>{display_label}</h2>
            <div class="confidence">Confidence: {confidence:.2f}%</div>
            <p class="description">{LABEL_DESCRIPTIONS[predicted_label]}</p>
        """, unsafe_allow_html=True)
        
        if predicted_label == 0:
            st.markdown('<div class="depressive-msg">💚 Your text shows signs of distress. Remember, it\'s okay to not be okay. Consider reaching out to someone you trust.</div>', unsafe_allow_html=True)
        elif predicted_label == 1:
            st.markdown('<div class="normal-msg">🤝 Your text appears balanced. Keep maintaining healthy emotional expression and self-care.</div>', unsafe_allow_html=True)
        elif predicted_label == 2:
            st.markdown('<div class="positive-msg">🌟 Your text reflects positive emotions. Great job maintaining mental wellness! Keep nurturing this positivity.</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_column:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📈 Class Probabilities")
        probability_rows = [{"Class": DISPLAY_LABELS[label], "Probability (%)": round(probability * 100, 2)} for label, probability in result["probabilities"].items()]
        probability_df = pd.DataFrame(probability_rows).set_index("Class")
        st.bar_chart(probability_df, color="#2B3034")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">🔬 SHAP Explanation</p>', unsafe_allow_html=True)
    st.caption(f"Word-level contributions for the predicted class: '{display_label}'")

    try:
        with st.spinner("Generating SHAP explanation..."):
            shap_html = explain_text_interactive(cleaned_text, result["pred_index"])
            components.html(shap_html, height=480, scrolling=True)
    except Exception as error:
        st.warning(f"Prediction succeeded, but SHAP could not be generated: {error}")

    st.markdown("---")
    st.info("**Important:** This is a research model. It is not intended for diagnosis, risk assessment, or as a substitute for professional mental health advice. If you are experiencing persistent or severe distress, please consult a qualified mental health professional.")