from functools import lru_cache
import html


import numpy as np
import shap


from model import DISPLAY_LABELS, ID2LABEL
from model_loader import (
    get_tokenizer,
    predict_probabilities,
)
from settings import settings


@lru_cache(maxsize=1)
def get_explainer():
    tokenizer = get_tokenizer()

    masker = shap.maskers.Text(
        tokenizer
    )

    output_names = [
        DISPLAY_LABELS[ID2LABEL[index]]
        for index in range(3)
    ]

    return shap.Explainer(
        predict_probabilities,
        masker,
        algorithm="partition",
        output_names=output_names,
    )


def shorten_text_for_shap(text: str) -> str:
    tokenizer = get_tokenizer()

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )["input_ids"]

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    ).strip()


def explain_text(
    text: str,
    target_class: int,
) -> dict:
    explanation_text = shorten_text_for_shap(
        text
    )

    tokenizer = get_tokenizer()

    encoded = tokenizer(
        explanation_text,
        add_special_tokens=True,
    )

    feature_count = len(
        encoded["input_ids"]
    )

    max_evals = max(
        2 * feature_count + 1,
        100,
    )

    shap_values = get_explainer()(
        [explanation_text],
        max_evals=max_evals,
        batch_size=settings.shap_batch_size,
    )

    sample_values = np.asarray(
        shap_values.values[0]
    )

    if sample_values.ndim == 2:
        class_values = sample_values[
            :,
            target_class,
        ]
    else:
        class_values = sample_values

    raw_tokens = np.asarray(
        shap_values.data[0]
    ).tolist()

    tokens = []

    for token, score in zip(
        raw_tokens,
        class_values,
    ):
        cleaned_token = str(token).strip()

        if not cleaned_token:
            continue

        if cleaned_token in {
            "[CLS]",
            "[SEP]",
            "[PAD]",
        }:
            continue

        tokens.append(
            (
                cleaned_token,
                float(score),
            )
        )

    tokens.sort(
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    return {
        "text_used": explanation_text,
        "truncated": (
            explanation_text.strip()
            != text.strip()
        ),
        "tokens": tokens,
    }


def render_token_contributions(
    tokens,
) -> str:
    if not tokens:
        return (
            "<p>No tokens available for explanation.</p>"
        )

    max_absolute = max(
        abs(score)
        for _, score in tokens
    ) or 1.0

    token_chips = []

    for token, score in tokens:
        strength = min(
            abs(score) / max_absolute,
            1.0,
        )

        opacity = (
            0.16 + 0.64 * strength
        )

        if score >= 0:
            background = (
                f"rgba(98,124,140,"
                f"{opacity:.3f})"
            )
            border = "#627C8C"
            sign = "+"
        else:
            background = (
                f"rgba(233,60,53,"
                f"{opacity:.3f})"
            )
            border = "#E93C35"
            sign = ""

        token_chips.append(
            "<span class='shap-token' "
            f"style='background:{background};"
            f"border-color:{border}' "
            f"title='SHAP: {sign}{score:.4f}'>"
            f"{html.escape(token)}"
            "</span>"
        )

    return (
        "<div class='shap-wrap'>"
        + "".join(token_chips)
        + "</div>"
    )


def explain_text_interactive(
    text: str,
    target_class: int,
) -> str:
    """Generate SHAP interactive HTML visualization"""
    explanation_text = shorten_text_for_shap(text)

    tokenizer = get_tokenizer()

    encoded = tokenizer(
        explanation_text,
        add_special_tokens=True,
    )

    feature_count = len(
        encoded["input_ids"]
    )

    max_evals = max(
        2 * feature_count + 1,
        100,
    )

    shap_values = get_explainer()(
        [explanation_text],
        max_evals=max_evals,
        batch_size=settings.shap_batch_size,
    )

    sample_explanation = shap_values[0]

    
    if len(sample_explanation.values.shape) == 2:
        
        html_parts = []
        for class_idx in range(3):
            class_shap = sample_explanation[:, class_idx]
            class_name = DISPLAY_LABELS[ID2LABEL[class_idx]]
            plot = shap.plots.text(class_shap)
            if hasattr(plot, 'html'):
                html_str = plot.html
            elif hasattr(plot, 'data'):
                html_str = plot.data
            else:
                html_str = str(plot)
            html_parts.append(f"<div style='margin-bottom:30px;'><h3 style='color:#2B3034; border-bottom:2px solid #E93C35; padding-bottom:10px;'>Class: {class_name}</h3>{html_str}</div>")
        return "<div>" + "<hr style='border:1px solid #989398; margin:20px 0;'>".join(html_parts) + "</div>"
    else:
        # Single class
        plot = shap.plots.text(sample_explanation)
        if hasattr(plot, 'html'):
            return plot.html
        elif hasattr(plot, 'data'):
            return plot.data
        else:
            return str(plot)