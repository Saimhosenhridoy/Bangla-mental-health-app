from functools import lru_cache
from pathlib import Path

import torch

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from model import (
    HybridBanglaBERTClassifier,
    ID2LABEL,
)
from settings import settings


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_weight_path() -> Path:
    local_path = Path(settings.model_path)

    if local_path.exists():
        return local_path

    if settings.hf_model_repo.strip():
        downloaded_path = hf_hub_download(
            repo_id=settings.hf_model_repo.strip(),
            filename=settings.hf_model_file,
            token=settings.hf_token,
        )

        return Path(downloaded_path)

    raise FileNotFoundError(
        f"Model weight not found: {local_path}. "
        "Please place best_model.pth in the weights folder."
    )


def load_checkpoint(path: Path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


@lru_cache(maxsize=1)
def get_model_name() -> str:
    checkpoint = load_checkpoint(
        get_weight_path()
    )

    if isinstance(checkpoint, dict):
        config = checkpoint.get(
            "config",
            {},
        )

        return str(
            config.get(
                "model_name",
                settings.model_name,
            )
        )

    return settings.model_name


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        get_model_name()
    )


@lru_cache(maxsize=1)
def load_model():
    weight_path = get_weight_path()

    checkpoint = load_checkpoint(
        weight_path
    )

    if isinstance(checkpoint, dict):
        config = checkpoint.get(
            "config",
            {},
        )
    else:
        config = {}

    model = HybridBanglaBERTClassifier(
        model_name=str(
            config.get(
                "model_name",
                settings.model_name,
            )
        ),
        num_labels=3,
        dropout=float(
            config.get(
                "dropout",
                0.35,
            )
        ),
        class_weights=config.get(
            "class_weights",
            [1.0, 1.5, 1.3],
        ),
        focal_gamma=float(
            config.get(
                "focal_gamma",
                1.0,
            )
        ),
        label_smoothing=float(
            config.get(
                "label_smoothing",
                0.10,
            )
        ),
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )
    else:
        state_dict = checkpoint

    if any(
        key.startswith("module.")
        for key in state_dict
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(get_device())
    model.eval()

    return model


def predict_probabilities(texts):
    if isinstance(texts, str):
        texts = [texts]

    texts = [
        str(text)
        for text in list(texts)
    ]

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
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs["logits"],
            dim=-1,
        )

    return (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )


def predict_text(text: str) -> dict:
    probabilities = predict_probabilities(
        [text]
    )[0]

    pred_index = int(
        probabilities.argmax()
    )

    pred_label = ID2LABEL[pred_index]

    return {
        "pred_index": pred_index,
        "pred_label": pred_label,
        "confidence": float(
            probabilities[pred_index]
        ),
        "probabilities": {
            ID2LABEL[index]: float(value)
            for index, value
            in enumerate(probabilities)
        },
        "device": str(get_device()),
    }