from functools import lru_cache
import html

import numpy as np
import shap

from model import DISPLAY_LABELS, ID2LABEL
from model_loader import get_tokenizer, predict_probabilities
from settings import settings

MAX_SHAP_CHARS = 180
MAX_EVALS_CAP = 61
SHAP_MAX_LENGTH = 64


@lru_cache(maxsize=1)
def get_explainer():
    output_names = [DISPLAY_LABELS[ID2LABEL[i]] for i in range(3)]
    masker = shap.maskers.Text(r"\s+")
    return shap.Explainer(
        predict_probabilities,
        masker,
        algorithm="partition",
        output_names=output_names,
        max_evals=MAX_EVALS_CAP,
    )


def shorten_text_for_shap(text: str) -> str:
    text = " ".join(str(text).split())
    if len(text) > MAX_SHAP_CHARS:
        text = text[:MAX_SHAP_CHARS].rsplit(" ", 1)[0]
    tokenizer = get_tokenizer()
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=SHAP_MAX_LENGTH,
    )["input_ids"]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def explain_text(text: str, target_class: int) -> dict:
    explanation_text = shorten_text_for_shap(text)
    n_words = max(len(explanation_text.split()), 1)
    max_evals = min(2 * n_words + 1, MAX_EVALS_CAP)
    if max_evals % 2 == 0:
        max_evals += 1

    shap_values = get_explainer()(
        [explanation_text],
        max_evals=max_evals,
        batch_size=getattr(settings, "shap_batch_size", 16),
    )

    sample_values = np.asarray(shap_values.values[0])
    if sample_values.ndim == 2:
        class_values = sample_values[:, target_class]
    else:
        class_values = sample_values

    raw_tokens = np.asarray(shap_values.data[0]).tolist()
    tokens = []
    for token, score in zip(raw_tokens, class_values):
        cleaned = str(token).strip()
        if not cleaned or cleaned in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        tokens.append((cleaned, float(score)))

    return {
        "text_used": explanation_text,
        "truncated": explanation_text.strip() != str(text).strip(),
        "tokens": tokens,
    }


def render_token_contributions(tokens) -> str:
    if not tokens:
        return "<p>No tokens available for explanation.</p>"
    max_absolute = max(abs(score) for _, score in tokens) or 1.0
    chips = []
    for token, score in tokens:
        strength = min(abs(score) / max_absolute, 1.0)
        opacity = 0.18 + 0.62 * strength
        if score >= 0:
            background = f"rgba(98,124,140,{opacity:.3f})"
            border = "#627C8C"
            sign = "+"
        else:
            background = f"rgba(233,60,53,{opacity:.3f})"
            border = "#E93C35"
            sign = ""
        chips.append(
            "<span class='shap-token' "
            f"style='background:{background};border-color:{border}' "
            f"title='SHAP: {sign}{score:.4f}'>"
            f"{html.escape(str(token))}</span>"
        )
    return "<div class='shap-wrap'>" + "".join(chips) + "</div>"


def _shap_text_html(explanation) -> str:
    try:
        plot = shap.plots.text(explanation, display=False)
    except TypeError:
        plot = shap.plots.text(explanation)
    if plot is None:
        raise RuntimeError("shap.plots.text returned nothing")
    if hasattr(plot, "data"):
        return plot.data
    if hasattr(plot, "html"):
        return plot.html
    return str(plot)


def explain_text_interactive(text: str, target_class: int) -> str:
    explanation_text = shorten_text_for_shap(text)
    n_words = max(len(explanation_text.split()), 1)
    max_evals = min(2 * n_words + 1, MAX_EVALS_CAP)
    if max_evals % 2 == 0:
        max_evals += 1

    shap_values = get_explainer()(
        [explanation_text],
        max_evals=max_evals,
        batch_size=getattr(settings, "shap_batch_size", 16),
    )

    sample = shap_values[0]

    try:
        shap_html = _shap_text_html(sample)
        return (
            "<div style='background:#ffffff;padding:12px;"
            "border-radius:12px;overflow:auto;'>"
            f"{shap_html}</div>"
        )
    except Exception:
        parts = []
        values = np.asarray(sample.values)
        n_classes = values.shape[1] if values.ndim == 2 else 1
        for class_idx in range(min(3, n_classes)):
            class_name = DISPLAY_LABELS[ID2LABEL[class_idx]]
            tokens = explain_text(text, class_idx)["tokens"]
            parts.append(
                f"<h3 style='color:#2B3034;border-bottom:2px solid #E93C35;'>"
                f"{class_name}</h3>"
                f"{render_token_contributions(tokens)}"
            )
        return (
            "<div style='background:#ffffff;padding:12px;'>"
            + "".join(parts)
            + "</div>"
        )