"""
Multi-task model for CEFR scoring + error detection.

Architecture:
    DeBERTa-v3 Encoder (shared) - Better than RoBERTa
        ↓
    ├─→ CEFR Score Head (regression)
    ├─→ Error Span Head (token classification)
    └─→ Error Type Head (multi-label classification)

Note: This model will be trained on Modal GPU, not locally.
"""

import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config


class FeedbackModel(nn.Module):
    """Multi-task model for essay feedback."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",  # Better than RoBERTa
        num_error_types: int = 5,  # grammar, vocab, mechanics, fluency, other
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared encoder - DeBERTa-v3
        self.encoder = DebertaV2Model.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for base

        # Task 1: CEFR Score Prediction (regression)
        self.cefr_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Task 2: Error Span Detection (token classification)
        # Output: [B-ERROR, I-ERROR, O] for each token
        self.span_head = nn.Linear(hidden_size, 3)

        # Task 3: Error Type Distribution (multi-label classification)
        # Output: probability for each error category
        self.error_type_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_error_types),
        )

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
            Dictionary with all task outputs
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = sequence_output[:, 0]  # CLS token [batch, hidden]

        # Task 1: CEFR score (use CLS token)
        cefr_score = self.cefr_head(pooled_output)  # [batch, 1]

        # Task 2: Error spans (all tokens)
        span_logits = self.span_head(sequence_output)  # [batch, seq_len, 3]

        # Task 3: Error types (use CLS token)
        error_type_logits = self.error_type_head(pooled_output)  # [batch, num_types]

        result = {
            "cefr_score": cefr_score.squeeze(-1),  # [batch]
            "span_logits": span_logits,
            "error_type_logits": error_type_logits,
        }

        # Optionally include attention weights for heatmap visualization
        if output_attentions:
            result["attentions"] = outputs.attentions

        return result


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning."""

    def __init__(
        self,
        cefr_weight: float = 1.0,
        span_weight: float = 0.5,
        error_type_weight: float = 0.3,
    ):
        super().__init__()
        self.cefr_weight = cefr_weight
        self.span_weight = span_weight
        self.error_type_weight = error_type_weight

        # Loss functions
        self.cefr_loss_fn = nn.MSELoss()
        self.span_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.error_type_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute weighted multi-task loss.

        Args:
            predictions: Model outputs
            targets: Ground truth labels

        Returns:
            (total_loss, loss_dict)
        """
        # Task 1: CEFR regression loss
        cefr_loss = self.cefr_loss_fn(predictions["cefr_score"], targets["cefr_score"])

        # Task 2: Span detection loss
        span_loss = self.span_loss_fn(
            predictions["span_logits"].view(-1, 3), targets["span_labels"].view(-1)
        )

        # Task 3: Error type loss
        error_type_loss = self.error_type_loss_fn(
            predictions["error_type_logits"], targets["error_type_labels"]
        )

        # Weighted combination
        total_loss = (
            self.cefr_weight * cefr_loss
            + self.span_weight * span_loss
            + self.error_type_weight * error_type_loss
        )

        # Return individual losses for monitoring
        loss_dict = {
            "total": total_loss.item(),
            "cefr": cefr_loss.item(),
            "span": span_loss.item(),
            "error_type": error_type_loss.item(),
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    print("✅ FeedbackModel module loaded")
    print("   Model: microsoft/deberta-v3-base")
    print("   Training will happen on Modal GPU")
    print("\nTo train:")
    print("  modal run scripts/training/train-feedback-model.py")
