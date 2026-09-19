import time

import altair as alt
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


def scroll_to(element_id: str):
    components.html(
        f"""
        <script>
        const doc = window.parent.document;
        const el = doc.getElementById("{element_id}");
        if (el) {{
            el.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}
        </script>
        """,
        height=0,
    )


def show_loader(placeholder, message: str):
    placeholder.markdown(
        f"""
        <div class="app-loader">
          <div class="app-loader-card">
            {message}
            <br><span>Please wait</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(0.05)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Hind+Siliguri:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Hind Siliguri', sans-serif !important; }
.stApp { background: #B9C7D2; }
.block-container { max-width: 1040px; padding-top: 0.8rem; padding-bottom: 2rem; }

#MainMenu, header, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"],
[data-testid="stFooter"],
[data-testid="stSidebar"],
.stDeployButton,
button[kind="header"],
div[class*="viewerBadge"],
a[href*="streamlit.io"],
a[href*="streamlit.app"] {
  display: none !important;
  visibility: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
  height: 0 !important;
}

.hero {
  padding: 1.5rem 1.2rem; border-radius: 20px; background: #2B3034; color: #fff;
  border: 4px solid #627C8C; text-align: center; margin-bottom: 1rem;
  position: relative; overflow: hidden;
}
.hero::before { content: "🧠"; position: absolute; top: -24px; left: -20px; font-size: 8rem; opacity: 0.08; }
.hero::after { content: "💚"; position: absolute; bottom: -18px; right: -16px; font-size: 7rem; opacity: 0.08; }
.hero .emoji-row { font-size: 2.1rem; margin-bottom: 0.35rem; position: relative; z-index: 1; }
.hero h1 { margin: 0; font-size: 1.7rem; position: relative; z-index: 1; }
.hero p { color: #B9C7D2; margin: 0.45rem auto 0; max-width: 640px; font-size: 0.9rem; position: relative; z-index: 1; }

.section-header {
  font-size: 1.15rem; font-weight: 700; color: #2B3034;
  margin: 1.1rem 0 0.35rem; padding-bottom: 0.3rem; border-bottom: 4px solid #E93C35;
}
.caption-text { color: #2B3034; font-size: 0.85rem; margin: 0 0 0.6rem; opacity: 0.85; }

.result-card {
  height: auto;
  min-height: 0;
  max-height: none;
  border-radius: 16px;
  padding: 1.1rem 1.15rem;
  box-sizing: border-box;
  overflow: visible;
  background: #627C8C;
  color: #fff;
  border: 3px solid #989398;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}
.result-card h2 { margin: 0.25rem 0; font-size: 1.4rem; }
.result-card .eyebrow { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
.result-card .confidence { font-size: 1.02rem; font-weight: 700; margin-top: 0.35rem; }
.result-card .description { color: #B9C7D2; font-size: 0.88rem; margin-top: 0.45rem; }
.class-msg {
  margin-top: 0.6rem; padding: 0.6rem 0.75rem; border-radius: 10px;
  background: #2B3034; color: #B9C7D2; font-size: 0.84rem;
}

[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stAltairChart"] {
  background: #ffffff !important;
  border: 3px solid #627C8C !important;
  border-radius: 16px !important;
  height: 280px !important;
  min-height: 280px !important;
  max-height: 280px !important;
  padding: 8px !important;
  overflow: hidden !important;
}
[data-testid="stVegaLiteChart"] [data-testid="stToolbar"],
[data-testid="stArrowVegaLiteChart"] button,
[data-testid="stAltairChart"] button {
  display: none !important;
}

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
.example-container div.stButton > button *,
.example-container div.stButton > button p,
.example-container div.stButton > button span,
.example-container div.stButton > button div {
  background: #B9C7D2 !important;
  color: #000000 !important;
  -webkit-text-fill-color: #000000 !important;
  border: 2px solid #2B3034 !important;
}

.app-loader {
  position: fixed; inset: 0;
  background: rgba(43, 48, 52, 0.55);
  display: flex; align-items: center; justify-content: center;
  z-index: 999999;
}
.app-loader-card {
  min-width: 260px; background: #2B3034; color: #B9C7D2;
  border: 3px solid #627C8C; border-radius: 16px;
  padding: 1.15rem 1.4rem; text-align: center;
  box-shadow: 0 12px 40px rgba(43,48,52,0.45); font-weight: 700;
}
.app-loader-card span {
  display: inline-block; margin-top: 0.45rem;
  font-size: 0.85rem; font-weight: 500; color: #989398;
}
#prediction-result, #shap-result { scroll-margin-top: 12px; }
.creator-id { text-align: center; color: #2B3034; font-size: 0.85rem; margin-top: 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="emoji-row">🧠 💚 🤝</div>
  <h1>Bangla Mental Health Classifier</h1>
  <p>Bangla social-media text classification with SHAP explainability. Not a clinical diagnosis.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="example-container">', unsafe_allow_html=True)
st.markdown(
    '<p class="section-header" style="margin-top:0;font-size:1rem;color:#111313;">💡 Try an example</p>',
    unsafe_allow_html=True,
)


def use_example(example_text):
    st.session_state["mental_text"] = example_text


examples = [
    (
        "Depressive",
        "ধীরে ধীরে বেঁচে থাকার ইচ্ছা হারিয়ে যাচ্ছে।",
    ),
    (
        "Non_depressive",
        "আজ সারাদিন কাজ করেছি, এখন বিশ্রাম নিচ্ছি।",
    ),
    (
        "Positive",
        "আজ আমি খুব আনন্দিত, বন্ধুদের সঙ্গে সময় কাটিয়েছি।",
    ),
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

st.markdown('<p class="section-header">⌨️ Enter your text</p>', unsafe_allow_html=True)
user_text = st.text_area(
    "Enter your text",
    placeholder="বাংলায় আপনার ভাবনা লিখুন...",
    label_visibility="collapsed",
    key="mental_text",
    max_chars=settings.max_text_chars,
)
analyze_button = st.button("🔄️ Analyze Text", type="primary", use_container_width=True)

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

    loader = st.empty()
    try:
        show_loader(loader, "Analyzing text...")
        result = predict_text(cleaned_text)
    except Exception as error:
        loader.empty()
        st.error(f"Model could not be loaded: {error}")
        st.stop()
    loader.empty()

    predicted_label = result["pred_label"]
    display_label = DISPLAY_LABELS[predicted_label]
    confidence = result["confidence"] * 100
    short_msg = CLASS_MESSAGES.get(predicted_label, "")

    st.markdown('<div id="prediction-result"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="caption-text">Predicted class: <b>{display_label}</b> · '
        "Probability for Depressive, Non_depressive, and Positive.</p>",
        unsafe_allow_html=True,
    )

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
        probability_df = pd.DataFrame(probability_rows)
        chart = (
            alt.Chart(probability_df)
            .mark_bar(color="#2B3034", size=42)
            .encode(
                x=alt.X(
                    "Class:N",
                    sort=["Depressive", "Non_depressive", "Positive"],
                    title=None,
                ),
                y=alt.Y(
                    "Probability (%):Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title=None,
                ),
            )
            .properties(height=230)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, domain=False)
        )
        st.altair_chart(chart, use_container_width=True)

    scroll_to("prediction-result")

    st.markdown('<div id="shap-result"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">SHAP Explanation</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="caption-text">Word-level contributions for Depressive, Non_depressive, and Positive. '
        "Blue/teal pushes toward a class, red pushes away.</p>",
        unsafe_allow_html=True,
    )

    shap_loader = st.empty()
    try:
        show_loader(shap_loader, "Generating SHAP explanation...")
        shap_html = explain_text_interactive(cleaned_text, result["pred_index"])
        shap_loader.empty()
        components.html(shap_html, height=700, scrolling=True)
        scroll_to("shap-result")
    except Exception as error:
        shap_loader.empty()
        st.warning(f"Prediction succeeded, but SHAP could not be generated: {error}")

    st.info(
        "This is a research model. It is not for diagnosis or as a substitute for professional care."
    )

st.markdown(
    '<p class="creator-id">Created by '
    '<a href="https://github.com/Saimhosenhridoy" '
    'style="color:#2B3034;font-weight:700;text-decoration:none;">'
    "@Saimhosenhridoy</a></p>",
    unsafe_allow_html=True,
)