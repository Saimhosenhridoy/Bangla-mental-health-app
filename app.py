import pandas as pd
import streamlit as st

from explain import (
    explain_text,
    render_token_contributions,
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
    page_title=(
        f"{settings.app_name} | "
        "Bangla Mental Health AI"
    ),
    page_icon="🫶",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
        linear-gradient(
            145deg,
            #f5f3ff 0%,
            #f0fdfa 48%,
            #fff7ed 100%
        );
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.2rem;
        border-radius: 28px;
        background:
        linear-gradient(
            120deg,
            #312e81,
            #6d28d9 56%,
            #0f766e
        );
        color: white;
        box-shadow:
        0 18px 55px
        rgba(49,46,129,.22);
        margin-bottom: 1.25rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.7rem;
    }

    .hero p {
        color: #ede9fe;
        font-size: 1.08rem;
        max-width: 760px;
    }

    .result-card {
        border-radius: 22px;
        padding: 1.5rem;
        color: white;
        background:
        linear-gradient(
            120deg,
            #4338ca,
            #7c3aed
        );
        box-shadow:
        0 14px 35px
        rgba(76,29,149,.20);
    }

    .result-card h2 {
        margin: .2rem 0;
        font-size: 1.9rem;
    }

    .result-card p {
        color: #ede9fe;
    }

    .eyebrow {
        font-size: .85rem;
        opacity: .8;
    }

    .shap-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        line-height: 2.3;
        padding: 8px 0 14px;
    }

    .shap-token {
        display: inline-block;
        padding: 3px 9px;
        border: 1px solid;
        border-radius: 9px;
        color: #172033;
        font-weight: 600;
        cursor: help;
    }

    .legend {
        color: #475569;
        font-size: .92rem;
    }

    .dot {
        display: inline-block;
        width: 11px;
        height: 11px;
        border-radius: 50%;
        margin: 0 5px 0 12px;
    }

    div[data-testid="stTextArea"] textarea {
        background: rgba(255,255,255,.94);
        border-radius: 16px;
        border: 1px solid #c4b5fd;
        min-height: 180px;
        font-size: 1.05rem;
    }

    div.stButton > button {
        border: 0;
        border-radius: 13px;
        font-weight: 700;
        min-height: 48px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <section class="hero">
        <div style="font-size:2rem">🫶</div>
        <h1>মনের কথা</h1>
        <p>
            বাংলা লেখার ভাষাগত ধরন বিশ্লেষণ করে
            সম্ভাব্য মানসিক অবস্থার শ্রেণি এবং
            SHAP-ভিত্তিক ব্যাখ্যা দেখুন।
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("ব্যবহার নির্দেশিকা")

    st.markdown(
        """
        1. নিজের অনুভূতি নিয়ে বাংলা লেখা দিন  
        2. **লেখা বিশ্লেষণ করুন** চাপুন  
        3. Prediction, probability ও SHAP দেখুন
        """
    )

    st.divider()

    enable_shap = st.toggle(
        "SHAP ব্যাখ্যা তৈরি করুন",
        value=True,
    )

    st.caption(
        "SHAP চালু থাকলে CPU-তে "
        "কিছুটা বেশি সময় লাগতে পারে।"
    )

    st.divider()

    st.info(
        f"চলমান ডিভাইস: `{get_device()}`"
    )

    st.caption(
        "গবেষণা ও শিক্ষামূলক ব্যবহার • "
        "Clinical diagnostic tool নয়"
    )


st.markdown(
    "### আপনি এখন কেমন অনুভব করছেন?"
)

st.caption(
    "এক বা একাধিক বাংলা বাক্যে লিখুন। "
    "কোনো ব্যক্তিগত পরিচয় বা গোপন তথ্য লিখবেন না।"
)


def use_example(example_text):
    st.session_state["mental_text"] = (
        example_text
    )


examples = [
    (
        "মন খারাপের উদাহরণ",
        "কয়েকদিন ধরে আমার কোনো কাজে মন বসছে না, "
        "সবকিছু খুব অর্থহীন মনে হচ্ছে।",
    ),
    (
        "সাধারণ উদাহরণ",
        "আজ সারাদিন কাজ করেছি, এখন বাসায় ফিরে "
        "বিশ্রাম নিচ্ছি।",
    ),
    (
        "ইতিবাচক উদাহরণ",
        "আজ আমি খুব আনন্দিত, অনেকদিন পর বন্ধুদের "
        "সঙ্গে সুন্দর সময় কাটিয়েছি।",
    ),
]

example_columns = st.columns(3)

for column, example in zip(
    example_columns,
    examples,
):
    button_label, example_text = example

    with column:
        st.button(
            button_label,
            use_container_width=True,
            on_click=use_example,
            args=(example_text,),
        )


user_text = st.text_area(
    "বাংলা লেখা",
    placeholder=(
        "উদাহরণ: কিছুদিন ধরে কোনো কাজে "
        "মন বসছে না এবং নিজেকে খুব একা লাগছে..."
    ),
    label_visibility="collapsed",
    key="mental_text",
    max_chars=settings.max_text_chars,
)


analyze_button = st.button(
    "লেখা বিশ্লেষণ করুন",
    type="primary",
    use_container_width=True,
)


if analyze_button:
    cleaned_text = clean_text(
        user_text
    )

    if len(cleaned_text) < 5:
        st.error(
            "বিশ্লেষণের জন্য অন্তত একটি "
            "অর্থপূর্ণ বাংলা বাক্য লিখুন।"
        )
        st.stop()

    if contains_crisis_language(
        cleaned_text
    ):
        st.error(
            "এই লেখায় নিজের ক্ষতি বা জীবন শেষ করার "
            "ইঙ্গিত থাকতে পারে। আপনি যদি তাৎক্ষণিক "
            "ঝুঁকিতে থাকেন, একা থাকবেন না—নিকটস্থ "
            "জরুরি সেবা, বিশ্বস্ত মানুষ অথবা যোগ্য "
            "মানসিক স্বাস্থ্য পেশাজীবীর সঙ্গে "
            "এখনই যোগাযোগ করুন।"
        )

    try:
        with st.spinner(
            "AI model লেখাটি বিশ্লেষণ করছে..."
        ):
            result = predict_text(
                cleaned_text
            )

    except Exception as error:
        st.error(
            f"Model চালু করা যায়নি: {error}"
        )

        st.info(
            "weights/best_model.pth fileটি "
            "সঠিক স্থানে আছে কি না পরীক্ষা করুন।"
        )

        st.stop()

    predicted_label = result[
        "pred_label"
    ]

    display_label = DISPLAY_LABELS[
        predicted_label
    ]

    confidence = (
        result["confidence"] * 100
    )

    st.markdown("---")

    result_column, chart_column = (
        st.columns(
            [1.35, 1],
            gap="large",
        )
    )

    with result_column:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="eyebrow">
                    MODEL PREDICTION
                </div>

                <h2>{display_label}</h2>

                <div style="
                    font-size:1.35rem;
                    font-weight:700;
                ">
                    Confidence:
                    {confidence:.2f}%
                </div>

                <p>
                    {
                        LABEL_DESCRIPTIONS[
                            predicted_label
                        ]
                    }
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with chart_column:
        st.markdown(
            "#### শ্রেণিভিত্তিক সম্ভাবনা"
        )

        probability_rows = []

        for label, probability in result[
            "probabilities"
        ].items():
            probability_rows.append(
                {
                    "শ্রেণি": DISPLAY_LABELS[
                        label
                    ],
                    "সম্ভাবনা (%)": round(
                        probability * 100,
                        2,
                    ),
                }
            )

        probability_df = pd.DataFrame(
            probability_rows
        ).set_index("শ্রেণি")

        st.bar_chart(
            probability_df,
            color="#6d28d9",
        )

    if enable_shap:
        st.markdown(
            "### কোন শব্দগুলো ফলাফলে প্রভাব ফেলেছে?"
        )

        st.caption(
            f"SHAP ব্যাখ্যাটি predicted class—"
            f"‘{display_label}’—এর জন্য।"
        )

        try:
            with st.spinner(
                "SHAP দিয়ে শব্দের প্রভাব "
                "হিসাব করা হচ্ছে..."
            ):
                shap_result = explain_text(
                    cleaned_text,
                    result["pred_index"],
                )

            st.markdown(
                render_token_contributions(
                    shap_result["tokens"]
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="legend">
                    <span
                        class="dot"
                        style="background:#10b981">
                    </span>
                    Predicted class-এর পক্ষে

                    <span
                        class="dot"
                        style="background:#f43f5e">
                    </span>
                    Predicted class-এর বিপক্ষে
                </div>
                """,
                unsafe_allow_html=True,
            )

            if shap_result["truncated"]:
                st.caption(
                    f"দ্রুত ব্যাখ্যার জন্য প্রথম "
                    f"{settings.shap_max_tokens} token "
                    "ব্যবহার করা হয়েছে। Prediction "
                    "সম্পূর্ণ লেখার ওপর করা হয়েছে।"
                )

            top_tokens = shap_result[
                "tokens"
            ][:12]

            if top_tokens:
                token_rows = []

                for token, score in top_tokens:
                    token_rows.append(
                        {
                            "শব্দ/Token": token,
                            "SHAP প্রভাব": round(
                                score,
                                5,
                            ),
                            "দিক": (
                                "পক্ষে"
                                if score >= 0
                                else "বিপক্ষে"
                            ),
                        }
                    )

                with st.expander(
                    "সবচেয়ে প্রভাবশালী token দেখুন"
                ):
                    st.dataframe(
                        pd.DataFrame(token_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as error:
            st.warning(
                "Prediction সফল হয়েছে, কিন্তু "
                f"SHAP তৈরি করা যায়নি: {error}"
            )

    st.markdown("---")

    st.warning(
        "গুরুত্বপূর্ণ: এটি একটি research model। "
        "এটি রোগ নির্ণয়, ঝুঁকি নির্ধারণ বা চিকিৎসা "
        "পরামর্শের বিকল্প নয়। মানসিক কষ্ট দীর্ঘস্থায়ী "
        "বা তীব্র হলে যোগ্য মানসিক স্বাস্থ্য "
        "পেশাজীবীর সহায়তা নিন।"
    )


st.markdown("---")

st.caption(
    "মনের কথা • Hybrid BanglaBERT + SHAP • "
    "লেখা স্থায়ীভাবে সংরক্ষণ করা হয় না"
)