from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from model import BanglaHAFBERT, ID2LABEL
from settings import settings


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_weight_path() -> Path:
    if settings.hf_model_repo and settings.hf_model_repo.strip():
        downloaded_path = hf_hub_download(
            repo_id=settings.hf_model_repo.strip(),
            filename=settings.hf_model_file,
            token=settings.hf_token,
        )
        return Path(downloaded_path)

    if settings.model_path and settings.model_path.strip():
        local_path = Path(settings.model_path)
        if local_path.exists():
            return local_path

    raise FileNotFoundError("HF_MODEL_REPO must be set in Secrets")


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


@lru_cache(maxsize=1)
def get_model_name() -> str:
    try:
        checkpoint = load_checkpoint(get_weight_path())
        if isinstance(checkpoint, dict):
            config = checkpoint.get("config", {})
            return str(config.get("model_name", settings.model_name))
    except Exception:
        pass
    return settings.model_name


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(get_model_name())


@lru_cache(maxsize=1)
def load_model():
    checkpoint = load_checkpoint(get_weight_path())
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    model = BanglaHAFBERT(
        source=str(config.get("model_name", settings.model_name)),
        flags=config.get("flags", {"gate": False}),
        num_labels=3,
        dropout=float(config.get("dropout", 0.35)),
    )

    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=False)
    model.to(get_device())
    model.eval()
    return model


def predict_probabilities(texts):
    if isinstance(texts, str):
        texts = [texts]
    if hasattr(texts, "tolist"):
        texts = texts.tolist()
    texts = [str(text) for text in list(texts)]

    tokenizer = get_tokenizer()
    model = load_model()
    device = get_device()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=settings.max_length,
        return_tensors="pt",
    )
    encoded = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }

    with torch.inference_mode():
        outputs = model(**encoded)
        if "final_probs" in outputs and outputs["final_probs"] is not None:
            probabilities = outputs["final_probs"]
        else:
            probabilities = torch.softmax(outputs["logits"], dim=-1)

    return probabilities.detach().cpu().numpy()


def predict_text(text: str) -> dict:
    probabilities = predict_probabilities([text])[0]
    pred_index = int(probabilities.argmax())
    pred_label = ID2LABEL[pred_index]
    return {
        "pred_index": pred_index,
        "pred_label": pred_label,
        "confidence": float(probabilities[pred_index]),
        "probabilities": {
            ID2LABEL[index]: float(value)
            for index, value in enumerate(probabilities)
        },
        "device": str(get_device()),
    }
