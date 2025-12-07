"""
DeBERTa-v3-large model for Automated Essay Scoring with dimensional outputs.

Architecture:
    DeBERTa-v3-large Encoder (304M params)
        ↓
    Attention Pooling (learned weighted combination of all tokens)
        ↓
    Multi-Head Output:
        - TA Head (Task Achievement) → 0-9
        - CC Head (Coherence & Cohesion) → 0-9
        - Vocab Head (Vocabulary) → 0-9
        - Grammar Head (Grammar) → 0-9
        - CEFR Head (Ordinal Regression) → 2-9 + level
"""

import torch
import torch.nn as nn
from transformers import DebertaV2Model

from pooling import AttentionPooling


class CORNHead(nn.Module):
    """
    Ordinal regression head using CORN (Consistent Rank Logits).

    Predicts ordered thresholds: P(score > 2), P(score > 3), ..., P(score > 8)
    Better suited for CEFR levels which have natural ordering.
    """

    def __init__(self, hidden_size: int, num_thresholds: int = 7):
        super().__init__()
        self.num_thresholds = num_thresholds

        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.threshold_classifiers = nn.Linear(256, num_thresholds)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Compute ordinal logits [batch_size, num_thresholds]."""
        shared = self.shared(pooled_output)
        return self.threshold_classifiers(shared)

    def logits_to_score(self, logits: torch.Tensor, min_score: float = 2.0) -> torch.Tensor:
        """Convert ordinal logits to continuous score."""
        probs = torch.sigmoid(logits)
        return min_score + probs.sum(dim=1)


class DimensionHead(nn.Module):
    """Regression head for a single dimension (TA, CC, Vocab, Grammar)."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Predict dimension score [batch_size, 1]."""
        return self.head(pooled_output)


class DeBERTaAESModel(nn.Module):
    """
    DeBERTa-v3-large model for essay scoring with multi-head dimensional outputs.

    Outputs:
        - TA (Task Achievement): 0-9
        - CC (Coherence & Cohesion): 0-9
        - Vocab (Vocabulary): 0-9
        - Grammar: 0-9
        - CEFR: 2-9 (ordinal regression)
        - Overall: Computed from dimensions
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        num_cefr_thresholds: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load DeBERTa-v3-large encoder
        self.encoder = DebertaV2Model.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 1024 for large

        # Attention pooling over all tokens
        self.pooling = AttentionPooling(hidden_size, dropout=dropout)

        # 4 IELTS-style dimension heads
        self.ta_head = DimensionHead(hidden_size, dropout)  # Task Achievement
        self.cc_head = DimensionHead(hidden_size, dropout)  # Coherence & Cohesion
        self.vocab_head = DimensionHead(hidden_size, dropout)  # Vocabulary
        self.grammar_head = DimensionHead(hidden_size, dropout)  # Grammar

        # CEFR ordinal regression head (for Overall + CEFR level)
        self.cefr_head = CORNHead(hidden_size, num_cefr_thresholds)

        # Store config
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            output_attentions: Whether to return attention weights

        Returns:
            Dictionary with dimension scores and CEFR predictions
        """
        # Encode with DeBERTa
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Attention pooling
        pooled = self.pooling(hidden_states, attention_mask)  # [batch, hidden]

        # Dimension predictions (0-9 scale)
        ta_score = self.ta_head(pooled).squeeze(-1)  # [batch]
        cc_score = self.cc_head(pooled).squeeze(-1)  # [batch]
        vocab_score = self.vocab_head(pooled).squeeze(-1)  # [batch]
        grammar_score = self.grammar_head(pooled).squeeze(-1)  # [batch]

        # CEFR ordinal regression
        cefr_logits = self.cefr_head(pooled)  # [batch, num_thresholds]
        cefr_score = self.cefr_head.logits_to_score(cefr_logits)  # [batch]

        # Overall as average of dimensions (can also use learned weights)
        overall = (ta_score + cc_score + vocab_score + grammar_score) / 4

        result = {
            "ta": ta_score,
            "cc": cc_score,
            "vocab": vocab_score,
            "grammar": grammar_score,
            "cefr_logits": cefr_logits,
            "cefr_score": cefr_score,
            "overall": overall,
        }

        if output_attentions and hasattr(outputs, "attentions"):
            result["attentions"] = outputs.attentions

        return result


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task dimensional + CEFR training."""

    def __init__(
        self,
        dimension_weight: float = 1.0,
        cefr_weight: float = 0.5,
        num_thresholds: int = 7,
        min_cefr_score: float = 2.0,
    ):
        super().__init__()
        self.dimension_weight = dimension_weight
        self.cefr_weight = cefr_weight
        self.num_thresholds = num_thresholds
        self.min_cefr_score = min_cefr_score

        # Loss functions
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute weighted multi-task loss.

        Args:
            predictions: Model outputs
            targets: Ground truth labels (may have missing dimensions)

        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions["ta"].device)

        # Dimension losses (if available)
        for dim in ["ta", "cc", "vocab", "grammar"]:
            if dim in targets and targets[dim] is not None:
                mask = ~torch.isnan(targets[dim])
                if mask.any():
                    loss = self.mse(predictions[dim][mask], targets[dim][mask])
                    losses[dim] = loss.item()
                    total_loss = total_loss + self.dimension_weight * loss

        # CEFR ordinal loss (if available)
        if "cefr" in targets and targets["cefr"] is not None:
            mask = ~torch.isnan(targets["cefr"])
            if mask.any():
                # Convert continuous CEFR to ordinal labels
                cefr_targets = targets["cefr"][mask]
                cefr_logits = predictions["cefr_logits"][mask]

                threshold_values = torch.arange(
                    self.num_thresholds,
                    device=cefr_logits.device,
                    dtype=torch.float32,
                )
                threshold_values = self.min_cefr_score + threshold_values

                ordinal_labels = (cefr_targets.unsqueeze(1) > threshold_values.unsqueeze(0)).float()

                cefr_loss = self.bce(cefr_logits, ordinal_labels).mean()
                losses["cefr"] = cefr_loss.item()
                total_loss = total_loss + self.cefr_weight * cefr_loss

        losses["total"] = total_loss.item()
        return total_loss, losses


if __name__ == "__main__":
    print("✅ DeBERTaAESModel module loaded")
    print("   Model: microsoft/deberta-v3-large (304M params)")
    print("   Outputs: TA, CC, Vocab, Grammar (0-9) + CEFR (2-9)")
    print("\nTo train:")
    print("  modal run scripts/training/train-deberta-aes.py")
