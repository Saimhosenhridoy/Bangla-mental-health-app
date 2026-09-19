from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "Bangla Mental Health Classifier"
    model_path: str = ""
    model_name: str = "csebuetnlp/banglabert"
    hf_model_repo: str = "saimhosenhridoy/bangla-mental-health-model"
    hf_model_file: str = "best_model.pth"
    hf_token: str | None = None
    max_length: int = 160
    max_text_chars: int = 2000
    shap_max_tokens: int = 40
    shap_batch_size: int = 8

settings = Settings()