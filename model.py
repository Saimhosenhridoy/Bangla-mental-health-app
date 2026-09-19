import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


LABEL2ID = {
    "Depressive": 0,
    "Non_depressive": 1,
    "Positive": 2,
}

ID2LABEL = {value: key for key, value in LABEL2ID.items()}

DISPLAY_LABELS = {
    "Depressive": "Depressive",
    "Non_depressive": "Non_depressive",
    "Positive": "Positive",
}

LABEL_DESCRIPTIONS = {
    "Depressive": (
        "The text shows linguistic indicators associated with "
        "sadness, hopelessness, or depressive thoughts."
    ),
    "Non_depressive": (
        "The text does not show clear linguistic indicators "
        "of depression."
    ),
    "Positive": (
        "The text shows linguistic indicators of an optimistic "
        "or positive mental state."
    ),
}


class LabelSmoothingFocalLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        gamma=1.0,
        smoothing=0.10,
        reduction="mean",
    ):
        super().__init__()
        if alpha is None:
            alpha = [1.0, 1.5, 1.3]
        self.register_buffer(
            "alpha",
            torch.tensor(alpha, dtype=torch.float32),
        )
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.to(device=logits.device, dtype=torch.long)
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probs)
        one_hot = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        smoothed_labels = (
            (1.0 - self.smoothing) * one_hot
            + self.smoothing / num_classes
        )
        smoothed_ce = -(smoothed_labels * log_probs).sum(dim=-1)
        target_probabilities = probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
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
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, token_embeddings, attention_mask):
        scores = self.attention(token_embeddings).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attention_vector = torch.sum(token_embeddings * weights, dim=1)
        expanded_mask = attention_mask.unsqueeze(-1).float()
        mean_vector = torch.sum(
            token_embeddings * expanded_mask, dim=1
        ) / expanded_mask.sum(dim=1).clamp(min=1e-9)
        return (attention_vector + mean_vector) / 2.0


class HybridBanglaBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels=3,
        dropout=0.35,
        class_weights=None,
        focal_gamma=1.0,
        label_smoothing=0.10,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )
        hidden_dim = self.bert.config.hidden_size
        self.self_attention = SelfAttentionWithMeanPooling(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.criterion = LabelSmoothingFocalLoss(
            alpha=class_weights or [1.0, 1.5, 1.3],
            gamma=focal_gamma,
            smoothing=label_smoothing,
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self.self_attention(
            outputs.last_hidden_state,
            attention_mask,
        )
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}