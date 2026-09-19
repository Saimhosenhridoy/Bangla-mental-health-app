import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from explain import (
    explain_text,
    render_token_contributions,
    explain_text_interactive,
)
from model import (
    DISPLAY_LABELS,
    LABEL_DESCRIPTIONS,
)
from model_loader import (
    get_device,
    predict_text,
)
from settings import settings
from text_utils import (
    clean_text,
    contains_crisis_language,
)


st.set_page_config(
    page_title="Bangla Mental Health Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================
# Responsive Professional Theme CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Hind+Siliguri:wght@400;500;600;700&display=swap');

/* =========================
   Global Reset
   ========================= */
* {
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    margin: 0;
    padding: 0;
}

/* =========================
   Sidebar Hide
   ========================= */
#root > div:nth-child(1) > div > div > div > div > section > div {
    display: none !important;
}

button[kid="collapse-button"] {
    display: none !important;
}

div[data-testid="stSidebar"] {
    display: none !important;
}

/* =========================
   Footer Hide
   ========================= */
footer {
    visibility: hidden;
    display: none !important;
}

div[data-testid="stFooter"] {
    visibility: hidden;
    display: none !important;
}

div[data-testid="stSidebarFooter"],
footer[data-testid="stAppFooter"] {
    display: none !important;
}

/* =========================
   Background & Layout
   ========================= */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
    min-height: 100vh;
}

.block-container {
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* =========================
   Hero Section - Responsive
   ========================= */
.hero {
    padding: 2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0891b2 100%);
    color: white;
    box-shadow: 0 20px 60px rgba(79, 70, 229, 0.25);
    margin-bottom: 2rem;
    text-align: center;
}

.hero h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.hero p {
    color: #e0e7ff;
    font-size: 1rem;
    max-width: 650px;
    margin: 0.75rem auto 0;
    line-height: 1.6;
}

/* Mobile */
@media (max-width: 640px) {
    .hero {
        padding: 1.5rem;
    }
    .hero h1 {
        font-size: 1.6rem;
    }
    .hero p {
        font-size: 0.9rem;
    }
}

/* Tablet */
@media (min-width: 641px) and (max-width: 1024px) {
    .hero h1 {
        font-size: 1.9rem;
    }
    .hero p {
        font-size: 0.95rem;
    }
}

/* Desktop */
@media (min-width: 1025px) {
    .hero {
        padding: 2.5rem;
    }
    .hero h1 {
        font-size: 2.8rem;
    }
    .hero p {
        font-size: 1.1rem;
    }
}

/* =========================
   Result Card - Responsive
   ========================= */
.result-card {
    border-radius: 20px;
    padding: 1.5rem;
    color: white;
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
    box-shadow: 0 15px 40px rgba(109, 40, 217, 0.25);
    border: 2px solid rgba(255, 255, 255, 0.15);
}

.result-card h2 {
    margin: 0.3rem 0;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.result-card .eyebrow {
    font-size: 0.75rem;
    opacity: 0.85;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.result-card .confidence {
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 0.5rem;
}

.result-card .description {
    color: #e0e7ff;
    font-size: 0.9rem;
    margin-top: 0.75rem;
    line-height: 1.6;
}

/* Mobile */
@media (max-width: 640px) {
    .result-card {
        padding: 1.25rem;
    }
    .result-card h2 {
        font-size: 1.3rem;
    }
    .result-card .confidence {
        font-size: 0.95rem;
    }
    .result-card .description {
        font-size: 0.85rem;
    }
}

/* Tablet */
@media (min-width: 641px) and (max-width: 1024px) {
    .result-card h2 {
        font-size: 1.5rem;
    }
}

/* Desktop */
@media (min-width: 1025px) {
    .result-card {
        padding: 2rem;
    }
    .result-card h2 {
        font-size: 2.2rem;
    }
    .result-card .confidence {
        font-size: 1.4rem;
    }
    .result-card .description {
        font-size: 1rem;
    }
}

/* =========================
   SHAP Section - Responsive
   ========================= */
.shap-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    line-height: 2.4;
    padding: 12px 0 16px;
}

.shap-token {
    display: inline-block;
    padding: 4px 11px;
    border: 1px solid;
    border-radius: 10px;
    color: #1e293b;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: help;
    transition: transform 0.15s ease;
}

.shap-token:hover {
    transform: scale(1.05);
}

.legend {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 8px;
}

.dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin: 0 6px 0 14px;
    vertical-align: middle;
}

/* Mobile */
@media (max-width: 640px) {
    .shap-wrap {
        gap: 6px;
        line-height: 2;
    }
    .shap-token {
        padding: 3px 8px;
        font-size: 0.85rem;
    }
}

/* =========================
   Text Area - Responsive
   ========================= */
div[data-testid="stTextArea"] textarea {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    border: 2px solid #c7d2fe;
    min-height: 180px;
    font-size: 1rem;
    font-family: 'Hind Siliguri', sans-serif !important;
    line-height: 1.7;
    width: 100% !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
}

/* Mobile */
@media (max-width: 640px) {
    div[data-testid="stTextArea"] textarea {
        min-height: 150px;
        font-size: 0.95rem;
    }
}

/* =========================
   Button - Responsive
   ========================= */
div.stButton > button {
    border: 0;
    border-radius: 14px;
    font-weight: 700;
    min-height: 48px;
    font-size: 1rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    width: 100% !important;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(79, 70, 229, 0.35);
}

/* Mobile */
@media (max-width: 640px) {
    div.stButton > button {
        min-height: 44px;
        font-size: 0.95rem;
    }
}

/* =========================
   Example Buttons - Responsive
   ========================= */
.example-btn {
    background: white !important;
    border: 2px solid #e0e7ff !important;
    color: #4f46e5 !important;
    font-weight: 600 !important;
}

.example-btn:hover {
    background: #f1f5f9 !important;
    border-color: #6366f1 !important;
}

/* =========================
   Section Headers - Responsive
   ========================= */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e293b;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #6366f1;
}

/* Mobile */
@media (max-width: 640px) {
    .section-header {
        font-size: 1.05rem;
        margin: 1.25rem 0 0.5rem;
    }
}

/* =========================
   Probability Chart - Responsive
   ========================= */
.chart-container {
    background: white;
    border-radius: 16px;
    padding: 1.25rem;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
    border: 1px solid #e2e8f0;
}

/* Mobile */
@media (max-width: 640px) {
    .chart-container {
        padding: 1rem;
    }
}

/* =========================
   Columns - Responsive
   ========================= */
/* Mobile: stack columns vertically */
@media (max-width: 768px) {
    .stColumns > div:first-child {
        width: 100% !important;
        margin-bottom: 1.5rem;
    }
    .stColumns > div:last-child {
        width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================
# Hero Section
# =========================
st.markdown("""
<div class="hero">
    <div style="font-size:2.5rem; margin-bottom:0.5rem">🧠</div>
    <h1>Bangla Mental Health Classifier</h1>
    <p>
        Analyze Bengali text to detect potential mental health indicators 
        with AI-powered classification and SHAP-based explanations.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# Examples
# =========================
st.markdown('<p class="section-header">Try an example</p>', unsafe_allow_html=True)

def use_example(example_text):
    st.session_state["mental_text"] = example_text

examples = [
    (
        "Depressive",
        "কয়েকদিন ধরে আমার কোনো কাজে মন বসছে না, সবকিছু খুব অর্থহীন মনে হচ্ছে।",
    ),
    (
        "Normal",
        "আজ সারাদিন কাজ করেছি, এখন বাসায় ফিরে বিশ্রাম নিচ্ছি।",
    ),
    (
        "Positive",
        "আজ আমি খুব আনন্দিত, অনেকদিন পর বন্ধুদের সঙ্গে সুন্দর সময় কাটিয়েছি।",
    ),
]

example_columns = st.columns(3)

for column, example in zip(example_columns, examples):
    button_label, example_text = example
    with column:
        st.button(
            button_label,
            use_container_width=True,
            on_click=use_example,
            args=(example_text,),
            key=f"ex_{button_label}",
        )


# =========================
# Text Input
# =========================
st.markdown('<p class="section-header">Enter your text</p>', unsafe_allow_html=True)

user_text = st.text_area(
    "",
    placeholder="Write your thoughts in Bengali here...",
    label_visibility="collapsed",
    key="mental_text",
    max_chars=settings.max_text_chars,
)


analyze_button = st.button(
    "Analyze Text",
    type="primary",
    use_container_width=True,
)


# =========================
# Analysis
# =========================
if analyze_button:
    cleaned_text = clean_text(user_text)

    if len(cleaned_text) < 5:
        st.error("Please enter at least one meaningful Bengali sentence.")
        st.stop()

    if contains_crisis_language(cleaned_text):
        st.error(
            "This text may contain indicators of self-harm or suicidal thoughts. "
            "If you are in immediate distress, please reach out to a trusted person, "
            "emergency services, or a qualified mental health professional right away."
        )

    try:
        with st.spinner("Analyzing text..."):
            result = predict_text(cleaned_text)

    except Exception as error:
        st.error(f"Model could not be loaded: {error}")
        st.info("Please ensure weights/best_model.pth file is in the correct location.")
        st.stop()

    predicted_label = result["pred_label"]
    display_label = DISPLAY_LABELS[predicted_label]
    confidence = result["confidence"] * 100

    st.markdown("---")

    # =========================
    # Prediction Result
    # =========================
    st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)

    result_column, chart_column = st.columns([1.4, 1], gap="large")

    with result_column:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="eyebrow">Prediction</div>
                <h2>{display_label}</h2>
                <div class="confidence">Confidence: {confidence:.2f}%</div>
                <p class="description">
                    {LABEL_DESCRIPTIONS[predicted_label]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with chart_column:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Class Probabilities")

        probability_rows = []
        for label, probability in result["probabilities"].items():
            probability_rows.append({
                "Class": DISPLAY_LABELS[label],
                "Probability (%)": round(probability * 100, 2),
            })

        probability_df = pd.DataFrame(probability_rows).set_index("Class")
        st.bar_chart(probability_df, color="#6366f1")
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # SHAP Explanation
    # =========================
    st.markdown('<p class="section-header">SHAP Explanation</p>', unsafe_allow_html=True)
    st.caption(f"Word-level contributions for the predicted class: '{display_label}'")

    shap_mode = st.radio(
        "Visualization mode",
        ["Simple (Token highlights)", "Interactive (Colab-style)"],
        index=0,
    )

    try:
        with st.spinner("Generating SHAP explanation..."):
            if shap_mode == "Interactive (Colab-style)":
                shap_html = explain_text_interactive(
                    cleaned_text,
                    result["pred_index"],
                )
                
                components.html(
                    shap_html,
                    height=480,
                    scrolling=True
                )
            else:
                shap_result = explain_text(
                    cleaned_text,
                    result["pred_index"],
                )

                st.markdown(
                    render_token_contributions(shap_result["tokens"]),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="legend">
                        <span class="dot" style="background:#10b981"></span>
                        Supports predicted class

                        <span class="dot" style="background:#f43f5e"></span>
                        Opposes predicted class
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if shap_result["truncated"]:
                    st.caption(
                        f"First {settings.shap_max_tokens} tokens used for faster explanation. "
                        "Prediction was made on the full text."
                    )

                top_tokens = shap_result["tokens"][:12]

                if top_tokens:
                    token_rows = []
                    for token, score in top_tokens:
                        token_rows.append({
                            "Token": token,
                            "SHAP Score": round(score, 5),
                            "Direction": "Supports" if score >= 0 else "Opposes",
                        })

                    with st.expander("View most influential tokens"):
                        st.dataframe(
                            pd.DataFrame(token_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

    except Exception as error:
        st.warning(f"Prediction succeeded, but SHAP could not be generated: {error}")

    st.markdown("---")

    st.info(
        "**Important:** This is a research model. It is not intended for diagnosis, "
        "risk assessment, or as a substitute for professional mental health advice. "
        "If you are experiencing persistent or severe distress, please consult a "
        "qualified mental health professional."
    )