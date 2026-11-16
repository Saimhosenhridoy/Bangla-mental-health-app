import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Automatically bridge Streamlit Cloud Secrets into environment variables
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ[k.upper()] = str(v)
except Exception:
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Bangla Mental Health Classifier"
    model_path: str = ""
    model_name: str = "csebuetnlp/banglabert"
    hf_model_repo: str = "Saimhosenhridoy/BanglaHAF-BERT"
    hf_model_file: str = "best_model.pt"
    hf_token: str | None = None

    max_length: int = 160
    max_text_chars: int = 2000

    shap_max_tokens: int = 36
    shap_max_chars: int = 160
    shap_max_evals: int = 35
    shap_batch_size: int = 8


settings = Settings()
