import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import html
import numpy as np
import shap
from settings import settings
from text_utils import clean_text, contains_crisis_language


# =========================
# Model Definitions (inline)
# =========================
LABEL2ID = {"Depressive": 0, "Non_depressive": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DISPLAY_LABELS = {
    "Depressive": "Signs of Depression",
    "Non_depressive": "No Signs of Depression",
    "Positive": "Positive Mental State",
}
LABEL_DESCRIPTIONS = {
    "Depressive": "The text shows linguistic indicators associated with sadness, hopelessness, or depressive thoughts.",
    "Non_depressive": "The text does not show clear linguistic indicators of depression.",
    "Positive": "The text shows linguistic indicators of an optimistic or positive mental state.",
}


class LabelSmoothingFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, smoothing=0.10, reduction="mean"):
        super().__init__()
        if alpha is None:
            alpha = [1.0, 1.5, 1.3]
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.to(device=logits.device, dtype=torch.long)
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probs)
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        smoothed_labels = (1.0 - self.smoothing) * one_hot + self.smoothing / num_classes
        smoothed_ce = -(smoothed_labels * log_probs).sum(dim=-1)
        target_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha[targets]
        focal_factor = (1.0 - target_probabilities).pow(self.gamma)
        loss = alpha_t * focal_factor * smoothed_ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class SelfAttentionWithMeanPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, token_embeddings, attention_mask):
        scores = self.attention(token_embeddings).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attention_vector = torch.sum(token_embeddings * weights, dim=1)
        expanded_mask = attention_mask.unsqueeze(-1).float()
        mean_vector = torch.sum(token_embeddings * expanded_mask, dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-9)
        return (attention_vector + mean_vector) / 2.0


class HybridBanglaBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.35, class_weights=None, focal_gamma=1.0, label_smoothing=0.10):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)
        hidden_dim = self.bert.config.hidden_size
        self.self_attention = SelfAttentionWithMeanPooling(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.criterion = LabelSmoothingFocalLoss(alpha=class_weights or [1.0, 1.5, 1.3], gamma=focal_gamma, smoothing=label_smoothing)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.self_attention(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}


# =========================
# Model Loader (inline)
# =========================
@lru_cache(maxsize=1)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_weight_path():
    local_path = Path(settings.model_path)
    if local_path.exists():
        return local_path
    if settings.hf_model_repo.strip():
        return Path(hf_hub_download(repo_id=settings.hf_model_repo.strip(), filename=settings.hf_model_file, token=settings.hf_token))
    raise FileNotFoundError(f"Model weight not found: {local_path}")


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@lru_cache(maxsize=1)
def get_model_name():
    checkpoint = load_checkpoint(get_weight_path())
    if isinstance(checkpoint, dict):
        return str(checkpoint.get("config", {}).get("model_name", settings.model_name))
    return settings.model_name


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(get_model_name())


@lru_cache(maxsize=1)
def load_model():
    weight_path = get_weight_path()
    checkpoint = load_checkpoint(weight_path)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = HybridBanglaBERTClassifier(
        model_name=str(config.get("model_name", settings.model_name)),
        num_labels=3,
        dropout=float(config.get("dropout", 0.35)),
        class_weights=config.get("class_weights", [1.0, 1.5, 1.3]),
        focal_gamma=float(config.get("focal_gamma", 1.0)),
        label_smoothing=float(config.get("label_smoothing", 0.10)),
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(get_device())
    model.eval()
    return model


def predict_probabilities(texts):
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in list(texts)]
    tokenizer = get_tokenizer()
    model = load_model()
    device = get_device()
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=settings.max_length, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs["logits"], dim=-1)
    return probabilities.detach().cpu().numpy()


def predict_text(text):
    probabilities = predict_probabilities([text])[0]
    pred_index = int(probabilities.argmax())
    pred_label = ID2LABEL[pred_index]
    return {
        "pred_index": pred_index,
        "pred_label": pred_label,
        "confidence": float(probabilities[pred_index]),
        "probabilities": {ID2LABEL[i]: float(v) for i, v in enumerate(probabilities)},
        "device": str(get_device()),
    }


# =========================
# SHAP Explainer (inline)
# =========================
@lru_cache(maxsize=1)
def get_explainer():
    tokenizer = get_tokenizer()
    masker = shap.maskers.Text(tokenizer)
    output_names = [DISPLAY_LABELS[ID2LABEL[index]] for index in range(3)]
    return shap.Explainer(predict_probabilities, masker, algorithm="partition", output_names=output_names)


def shorten_text_for_shap(text):
    tokenizer = get_tokenizer()
    token_ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=settings.shap_max_tokens)["input_ids"]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def explain_text(text, target_class):
    explanation_text = shorten_text_for_shap(text)
    tokenizer = get_tokenizer()
    encoded = tokenizer(explanation_text, add_special_tokens=True)
    feature_count = len(encoded["input_ids"])
    max_evals = max(2 * feature_count + 1, 20)
    shap_values = get_explainer()([explanation_text], max_evals=max_evals, batch_size=settings.shap_batch_size)
    sample_values = np.asarray(shap_values.values[0])
    if sample_values.ndim == 2:
        class_values = sample_values[:, target_class]
    else:
        class_values = sample_values
    raw_tokens = np.asarray(shap_values.data[0]).tolist()
    tokens = []
    for token, score in zip(raw_tokens, class_values):
        cleaned_token = str(token).strip()
        if not cleaned_token or cleaned_token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        tokens.append((cleaned_token, float(score)))
    tokens.sort(key=lambda item: abs(item[1]), reverse=True)
    return {"text_used": explanation_text, "truncated": explanation_text.strip() != text.strip(), "tokens": tokens}


def render_token_contributions(tokens):
    if not tokens:
        return "<p>No tokens available for explanation.</p>"
    max_absolute = max(abs(score) for _, score in tokens) or 1.0
    token_chips = []
    for token, score in tokens:
        strength = min(abs(score) / max_absolute, 1.0)
        opacity = 0.16 + 0.64 * strength
        if score >= 0:
            background = f"rgba(16,185,129,{opacity:.3f})"
            border = "#10b981"
            sign = "+"
        else:
            background = f"rgba(244,63,94,{opacity:.3f})"
            border = "#f43f5e"
            sign = ""
        token_chips.append(f"<span class='shap-token' style='background:{background};border-color:{border}' title='SHAP: {sign}{score:.4f}'>{html.escape(token)}</span>")
    return "<div class='shap-wrap'>" + "".join(token_chips) + "</div>"


def explain_text_interactive(text, target_class):
    explanation_text = shorten_text_for_shap(text)
    shap_values = get_explainer()([explanation_text], max_evals=50, batch_size=settings.shap_batch_size)
    sample_explanation = shap_values[0]
    if len(sample_explanation.values.shape) == 2:
        sample_explanation = sample_explanation[:, target_class]
    shap_html = shap.plots.text(sample_explanation, display=False)
    return shap_html


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Bangla Mental Health Classifier", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Hind+Siliguri:wght@400;500;600;700&display=swap');
* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#root > div:nth-child(1) > div > div > div > div > section > div, button[kid="collapse-button"], div[data-testid="stSidebar"], footer, div[data-testid="stFooter"] { display: none !important; }
.stApp { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%); min-height: 100vh; }
.block-container { max-width: 1000px; padding-top: 2rem; padding-bottom: 3rem; }
.hero { padding: 2rem; border-radius: 24px; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0891b2 100%); color: white; box-shadow: 0 20px 60px rgba(79, 70, 229, 0.25); margin-bottom: 2rem; text-align: center; }
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 800; }
.hero p { color: #e0e7ff; font-size: 1rem; max-width: 650px; margin: 0.75rem auto 0; }
.result-card { border-radius: 20px; padding: 1.5rem; color: white; background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%); box-shadow: 0 15px 40px rgba(109, 40, 217, 0.25); border: 2px solid rgba(255, 255, 255, 0.15); }
.result-card h2 { margin: 0.3rem 0; font-size: 1.6rem; font-weight: 800; }
.result-card .eyebrow { font-size: 0.75rem; opacity: 0.85; font-weight: 600; text-transform: uppercase; }
.result-card .confidence { font-size: 1.1rem; font-weight: 700; margin-top: 0.5rem; }
.result-card .description { color: #e0e7ff; font-size: 0.9rem; margin-top: 0.75rem; }
.shap-wrap { display: flex; flex-wrap: wrap; gap: 10px; line-height: 2.4; padding: 12px 0 16px; }
.shap-token { display: inline-block; padding: 4px 11px; border: 1px solid; border-radius: 10px; color: #1e293b; font-weight: 600; cursor: help; }
.legend { color: #64748b; font-size: 0.9rem; margin-top: 8px; }
.dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin: 0 6px 0 14px; vertical-align: middle; }
div[data-testid="stTextArea"] textarea { background: rgba(255, 255, 255, 0.95); border-radius: 16px; border: 2px solid #c7d2fe; min-height: 180px; font-size: 1rem; font-family: 'Hind Siliguri', sans-serif !important; }
div.stButton > button { border: 0; border-radius: 14px; font-weight: 700; min-height: 48px; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25); }
.section-header { font-size: 1.2rem; font-weight: 700; color: #1e293b; margin: 1.5rem 0 0.75rem; padding-bottom: 0.5rem; border-bottom: 3px solid #6366f1; }
.chart-container { background: white; border-radius: 16px; padding: 1.25rem; box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06); border: 1px solid #e2e8f0; }
@media (max-width: 768px) { .stColumns > div:first-child, .stColumns > div:last-child { width: 100% !important; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div style="font-size:2.5rem; margin-bottom:0.5rem">🧠</div>
    <h1>Bangla Mental Health Classifier</h1>
    <p>Analyze Bengali text to detect potential mental health indicators with AI-powered classification and SHAP-based explanations.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="section-header">Try an example</p>', unsafe_allow_html=True)

def use_example(example_text):
    st.session_state["mental_text"] = example_text

examples = [
    ("Depressive", "কয়েকদিন ধরে আমার কোনো কাজে মন বসছে না, সবকিছু খুব অর্থহীন মনে হচ্ছে।"),
    ("Normal", "আজ সারাদিন কাজ করেছি, এখন বাসায় ফিরে বিশ্রাম নিচ্ছি।"),
    ("Positive", "আজ আমি খুব আনন্দিত, অনেকদিন পর বন্ধুদের সঙ্গে সুন্দর সময় কাটিয়েছি।"),
]

example_columns = st.columns(3)
for column, example in zip(example_columns, examples):
    button_label, example_text = example
    with column:
        st.button(button_label, use_container_width=True, on_click=use_example, args=(example_text,), key=f"ex_{button_label}")

st.markdown('<p class="section-header">Enter your text</p>', unsafe_allow_html=True)
user_text = st.text_area("", placeholder="Write your thoughts in Bengali here...", label_visibility="collapsed", key="mental_text", max_chars=settings.max_text_chars)

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
        st.info("Please ensure weights/best_model.pth file is in the correct location.")
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
        st.bar_chart(probability_df, color="#6366f1")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">SHAP Explanation</p>', unsafe_allow_html=True)
    st.caption(f"Word-level contributions for the predicted class: '{display_label}'")

    shap_mode = st.radio("Visualization mode", ["Simple (Token highlights)", "Interactive (Colab-style)"], index=0)

    try:
        with st.spinner("Generating SHAP explanation..."):
            if shap_mode == "Interactive (Colab-style)":
                shap_html = explain_text_interactive(cleaned_text, result["pred_index"])
                components.html(shap_html, height=480, scrolling=True)
            else:
                shap_result = explain_text(cleaned_text, result["pred_index"])
                st.markdown(render_token_contributions(shap_result["tokens"]), unsafe_allow_html=True)
                st.markdown("""
                <div class="legend">
                    <span class="dot" style="background:#10b981"></span> Supports predicted class
                    <span class="dot" style="background:#f43f5e"></span> Opposes predicted class
                </div>
                """, unsafe_allow_html=True)
                if shap_result["truncated"]:
                    st.caption(f"First {settings.shap_max_tokens} tokens used for faster explanation. Prediction was made on the full text.")
                top_tokens = shap_result["tokens"][:12]
                if top_tokens:
                    token_rows = [{"Token": token, "SHAP Score": round(score, 5), "Direction": "Supports" if score >= 0 else "Opposes"} for token, score in top_tokens]
                    with st.expander("View most influential tokens"):
                        st.dataframe(pd.DataFrame(token_rows), use_container_width=True, hide_index=True)
    except Exception as error:
        st.warning(f"Prediction succeeded, but SHAP could not be generated: {error}")

    st.markdown("---")
    st.info("**Important:** This is a research model. It is not intended for diagnosis, risk assessment, or as a substitute for professional mental health advice. If you are experiencing persistent or severe distress, please consult a qualified mental health professional.")
