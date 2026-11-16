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


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden, attention_heads=4, attention_dropout=0.10):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(attention_dropout),
            nn.Linear(hidden, attention_heads),
        )
        self.projection = nn.Linear(
            hidden * attention_heads, hidden
        )

    def forward(self, x, mask):
        scores = self.attention(x.float())
        scores = scores.masked_fill(
            mask.unsqueeze(-1) == 0, -1e4
        )
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum(
            "bsh,bsd->bhd", weights, x.float()
        )
        return self.projection(pooled.flatten(1))


class MultiScaleCNN(nn.Module):
    def __init__(self, hidden, cnn_channels=128, cnn_kernels=(2, 3, 5)):
        super().__init__()
        self.kernels = cnn_kernels
        self.convs = nn.ModuleList([
            nn.Conv1d(
                hidden,
                cnn_channels,
                kernel_size=k,
            )
            for k in self.kernels
        ])
        self.projection = nn.Linear(
            cnn_channels * len(self.kernels),
            hidden,
        )

    def forward(self, x, attention_mask):
        z = x.float().transpose(1, 2)
        lengths = attention_mask.sum(dim=1)
        all_features = []
        for conv, kernel_size in zip(self.convs, self.kernels):
            z_pad = F.pad(
                z,
                (0, max(0, kernel_size - z.shape[-1])),
            )
            feature_map = F.gelu(conv(z_pad))
            valid_lengths = torch.clamp(
                lengths - kernel_size + 1,
                min=1,
                max=feature_map.shape[-1],
            )
            positions = torch.arange(
                feature_map.shape[-1],
                device=x.device,
            ).unsqueeze(0)
            invalid = positions >= valid_lengths.unsqueeze(1)
            feature_map = feature_map.masked_fill(
                invalid.unsqueeze(1), -1e4
            )
            all_features.append(
                feature_map.max(dim=-1).values
            )
        concatenated = torch.cat(all_features, dim=1)
        return self.projection(concatenated)


class ClassSpecificAttention(nn.Module):
    def __init__(self, hidden, n_classes=3, attention_dropout=0.10):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(attention_dropout),
            nn.Linear(hidden, n_classes),
        )
        self.class_vectors = nn.Parameter(
            torch.empty(n_classes, hidden)
        )
        self.class_bias = nn.Parameter(
            torch.zeros(n_classes)
        )
        nn.init.xavier_uniform_(self.class_vectors)
        self.scale = hidden ** -0.5

    def forward(self, token_hidden, attention_mask):
        scores = self.attention(token_hidden.float())
        scores = scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0,
            -1e4,
        )
        weights = torch.softmax(scores, dim=1)
        class_representations = torch.einsum(
            "bsc,bsh->bch",
            weights,
            token_hidden.float(),
        )
        logits = (
            (
                class_representations
                * self.class_vectors.unsqueeze(0)
            ).sum(dim=-1)
            * self.scale
            + self.class_bias
        )
        return logits, weights, class_representations


class FeatureFusion(nn.Module):
    def __init__(self, hidden, n, gated=False, dropout=0.35):
        super().__init__()
        self.gated = gated
        if gated:
            self.gate = nn.Sequential(
                nn.Linear(hidden * n, hidden),
                nn.GELU(),
                nn.Linear(hidden, n),
            )
        self.refine = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, reps):
        if self.gated:
            weights = self.gate(torch.cat(reps, dim=-1)).softmax(-1)
        else:
            weights = torch.full(
                (reps[0].shape[0], len(reps)),
                1.0 / len(reps),
                device=reps[0].device,
            )
        fused = (torch.stack(reps, 1) * weights.unsqueeze(-1)).sum(1)
        return self.norm(fused + self.refine(fused)), weights


class BanglaHAFBERT(nn.Module):
    """
    BanglaHAF-BERT: Proposed Hybrid Attention Fusion Architecture.
    Combines:
      - BanglaBERT Encoder (csebuetnlp/banglabert)
      - Multi-Head Attention Pooling
      - Multi-Scale 1D-CNN (kernels: 2, 3, 5)
      - Equal/Gated 4-way Feature Fusion (Attention, Mean, Max, CNN)
      - Class-Specific Token Attention (Evidence scaling)
      - Hierarchical Classification Heads (Depressive, Positive, Adaptive Gate)
      - Supervised Contrastive Projection Head
    """
    def __init__(
        self,
        source="csebuetnlp/banglabert",
        flags=None,
        num_labels=3,
        dropout=0.35,
        attention_heads=4,
        attention_dropout=0.10,
        cnn_channels=128,
        cnn_kernels=(2, 3, 5),
        projection_dim=64,
        **kwargs,
    ):
        super().__init__()
        if flags is None:
            flags = {"gate": False}  # B0_PROPOSED_EF
        self.flags = flags

        self.encoder = AutoModel.from_pretrained(
            source,
            low_cpu_mem_usage=True,
        )
        hidden = self.encoder.config.hidden_size

        self.attention_pool = MultiHeadAttentionPooling(
            hidden,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
        )

        if flags.get("cnn", True):
            self.cnn = MultiScaleCNN(
                hidden,
                cnn_channels=cnn_channels,
                cnn_kernels=cnn_kernels,
            )

        if flags.get("class_attention", True):
            self.class_attention = ClassSpecificAttention(
                hidden,
                n_classes=num_labels,
                attention_dropout=attention_dropout,
            )
            self.class_evidence_log_scale = nn.Parameter(torch.tensor(-0.7))

        n_reps = 4 if flags.get("cnn", True) else 3
        self.fusion = FeatureFusion(
            hidden,
            n_reps,
            gated=flags.get("gate", False),
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.main_head = nn.Linear(hidden, num_labels)

        if flags.get("hier", True):
            self.depressive_head = nn.Linear(hidden, 1)
            self.positive_head = nn.Linear(hidden, 1)
            self.hierarchical_gate = nn.Linear(hidden, 1)

        self.projection = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim),
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        tokens = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state.float()

        mask = attention_mask.unsqueeze(-1).float()
        att = self.attention_pool(tokens, attention_mask)
        mean = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1)
        maximum = tokens.masked_fill(mask == 0, -1e4).max(1).values
        reps = [att, mean, maximum]

        if self.flags.get("cnn", True):
            reps.append(self.cnn(tokens, attention_mask))

        fused, gates = self.fusion(reps)
        embeddings = F.normalize(self.projection(fused), dim=-1)
        fused = self.dropout(fused)
        main = self.main_head(fused)

        cls = None
        attention = None
        scale = torch.tensor(0.0, device=fused.device)

        if self.flags.get("class_attention", True):
            cls, attention, _ = self.class_attention(tokens, attention_mask)
            scale = self.class_evidence_log_scale.sigmoid()
            main = main + scale * cls

        mp = main.softmax(-1)
        dep = None
        pos = None
        hp = mp
        lamb = torch.zeros((len(mp), 1), device=mp.device)

        if self.flags.get("hier", True):
            dep = self.depressive_head(fused).squeeze(-1)
            pos = self.positive_head(fused).squeeze(-1)
            pd = dep.sigmoid()
            pp = pos.sigmoid()
            hp = torch.stack([pd, (1 - pd) * (1 - pp), (1 - pd) * pp], -1)
            lamb = self.hierarchical_gate(fused).sigmoid()

        final_probs = (1 - lamb) * mp + lamb * hp

        return {
            "logits": main,
            "final_probs": final_probs,
            "main_probs": mp,
            "hierarchical_probs": hp,
            "main_logits": main,
            "class_logits": cls,
            "dep_logits": dep,
            "pos_logits": pos,
            "embeddings": embeddings,
            "fusion_gates": gates,
            "hierarchical_lambda": lamb,
            "class_attention_weights": attention,
            "class_evidence_scale": scale,
        }


# Aliases for compatibility
HNAHSBERTV7 = BanglaHAFBERT
HybridBanglaBERTClassifier = BanglaHAFBERT
