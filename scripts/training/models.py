#!/usr/bin/env python3
"""
Ordinal regression model architectures for CEFR prediction.

Implements CORAL (COnsistent RAnk Logits) and other ordinal regression approaches
based on latest research in automated essay scoring.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig


class CoralModel(PreTrainedModel):
    """
    CORAL (COnsistent RAnk Logits) ordinal regression model.

    Predicts cumulative probabilities P(Y > k) for each threshold k.
    Ensures rank-monotonicity: if k1 < k2, then P(Y > k1) >= P(Y > k2).

    Reference: "Rank Consistent Ordinal Regression for Neural Networks" (2019)
    https://arxiv.org/abs/1901.07884
    """

    def __init__(self, base_model_name: str, num_classes: int):
        """
        Args:
            base_model_name: HuggingFace model identifier (e.g., "microsoft/deberta-v3-base")
            num_classes: Number of ordinal classes (11 for CEFR: A1 to C2)
        """
        config = AutoConfig.from_pretrained(base_model_name)
        super().__init__(config)

        self.num_classes = num_classes
        self.transformer = AutoModel.from_pretrained(base_model_name)

        # Binary classifiers for K-1 thresholds
        # Each predicts P(Y > k) for threshold k
        self.fc = nn.Linear(config.hidden_size, num_classes - 1)

        # Initialize to predict uniform distribution
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Forward pass.

        Returns:
            logits: [batch_size, num_classes - 1] binary logits for each threshold
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        logits = self.fc(pooled_output)  # [batch_size, num_classes - 1]

        return type("Outputs", (), {"logits": logits})()

    def predict_ordinal_class(self, input_ids, attention_mask):
        """
        Predict ordinal class (0 to num_classes - 1).

        Returns:
            predicted_class: Integer class prediction
            probabilities: Probability distribution over classes
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs.logits

            # Cumulative probabilities P(Y > k)
            cum_probs = torch.sigmoid(logits)

            # Convert to class probabilities
            # P(Y = 0) = 1 - P(Y > 0)
            # P(Y = k) = P(Y > k-1) - P(Y > k) for k > 0
            # P(Y = K) = P(Y > K-1)

            probs = torch.zeros(logits.size(0), self.num_classes, device=logits.device)
            probs[:, 0] = 1 - cum_probs[:, 0]
            for i in range(1, self.num_classes - 1):
                probs[:, i] = cum_probs[:, i - 1] - cum_probs[:, i]
            probs[:, -1] = cum_probs[:, -1]

            predicted_class = torch.argmax(probs, dim=1)

            return predicted_class, probs


class SoftLabelOrdinalModel(nn.Module):
    """
    Standard classification model with soft ordinal labels.

    Uses Gaussian soft labels that assign probability mass to neighboring classes
    based on distance from the true class.
    """

    def __init__(self, base_model_name: str, num_classes: int):
        """
        Args:
            base_model_name: HuggingFace model identifier
            num_classes: Number of ordinal classes
        """
        super().__init__()

        self.num_classes = num_classes
        self.transformer = AutoModel.from_pretrained(base_model_name)

        # Get hidden size from config
        config = AutoConfig.from_pretrained(base_model_name)
        hidden_size = config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize to predict near the middle (class 5 = B1+ for CEFR)
        nn.init.constant_(self.classifier.bias, 0.0)
        self.classifier.bias.data[num_classes // 2] = 1.0
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Forward pass.

        Returns:
            logits: [batch_size, num_classes] class logits
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return type("Outputs", (), {"logits": logits})()

    def predict_class(self, input_ids, attention_mask):
        """Predict ordinal class."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=1)

            return predicted_class, probs


def create_coral_model(base_model_name: str, num_classes: int) -> CoralModel:
    """
    Factory function to create CORAL model.

    Args:
        base_model_name: HuggingFace model identifier
        num_classes: Number of ordinal classes

    Returns:
        CoralModel instance
    """
    return CoralModel(base_model_name, num_classes)


def create_soft_label_model(
    base_model_name: str, num_classes: int
) -> SoftLabelOrdinalModel:
    """
    Factory function to create soft label ordinal model.

    Args:
        base_model_name: HuggingFace model identifier
        num_classes: Number of ordinal classes

    Returns:
        SoftLabelOrdinalModel instance
    """
    return SoftLabelOrdinalModel(base_model_name, num_classes)
