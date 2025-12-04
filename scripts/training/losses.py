#!/usr/bin/env python3
"""
Custom loss functions for ordinal regression in automated essay scoring.

Implements research-backed loss functions optimized for ordinal classification tasks.
"""

import torch
import torch.nn.functional as F


def coral_loss(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    CORAL (COnsistent RAnk Logits) loss for ordinal regression.

    Converts ordinal labels into K-1 binary classification tasks and ensures
    rank-monotonicity in predictions.

    Reference: "Rank Consistent Ordinal Regression for Neural Networks"
    https://arxiv.org/abs/1901.07884

    Args:
        logits: [batch_size, num_classes - 1] binary logits for each threshold
        labels: [batch_size] integer class labels (0 to num_classes - 1)
        num_classes: Total number of ordinal classes

    Returns:
        loss: Scalar loss value
    """
    # Create binary labels for each threshold
    # For class k: [1, 1, ..., 1, 0, 0, ..., 0] (k ones, followed by zeros)
    levels = torch.arange(num_classes - 1, dtype=torch.float32, device=labels.device)

    # Expand labels to [batch_size, 1] for broadcasting
    labels = labels.unsqueeze(1).float()

    # binary_labels[i, j] = 1 if labels[i] > j, else 0
    binary_labels = (labels > levels).float()

    # Binary cross-entropy for each threshold
    loss = F.binary_cross_entropy_with_logits(logits, binary_labels, reduction="mean")

    return loss


def soft_label_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int, sigma: float = 1.0
) -> torch.Tensor:
    """
    Cross-entropy loss with Gaussian soft labels for ordinal regression.

    Creates soft label distributions centered at the true class with
    probability mass assigned to neighboring classes based on distance.

    Args:
        logits: [batch_size, num_classes] class logits
        labels: [batch_size] integer class labels
        num_classes: Total number of classes
        sigma: Standard deviation of Gaussian distribution (controls softness)

    Returns:
        loss: Scalar loss value
    """
    logits.size(0)

    # Create class indices [0, 1, 2, ..., num_classes - 1]
    classes = torch.arange(num_classes, dtype=torch.float32, device=logits.device)

    # Expand for broadcasting
    labels_expanded = labels.unsqueeze(1).float()  # [batch_size, 1]
    classes_expanded = classes.unsqueeze(0)  # [1, num_classes]

    # Gaussian distribution around true class
    # soft_labels[i, j] = exp(-((j - labels[i])^2) / (2 * sigma^2))
    distances = (classes_expanded - labels_expanded) ** 2
    soft_labels = torch.exp(-distances / (2 * sigma**2))

    # Normalize to sum to 1
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)

    # Cross-entropy with soft labels
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_labels * log_probs).sum(dim=1).mean()

    return loss


def focal_loss(
    logits: torch.Tensor, labels: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """
    Focal loss for addressing class imbalance.

    Focuses training on hard examples by down-weighting easy examples.
    Useful when some CEFR levels have very few samples.

    Reference: "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002

    Args:
        logits: [batch_size, num_classes] class logits
        labels: [batch_size] integer class labels
        alpha: Weighting factor (0-1)
        gamma: Focusing parameter (>= 0, typically 2.0)

    Returns:
        loss: Scalar loss value
    """
    # Standard cross-entropy
    ce_loss = F.cross_entropy(logits, labels, reduction="none")

    # p_t = probability of true class
    p_t = torch.exp(-ce_loss)

    # Focal loss: FL = -alpha * (1 - p_t)^gamma * log(p_t)
    focal_loss = alpha * (1 - p_t) ** gamma * ce_loss

    return focal_loss.mean()


def cdw_ce_loss(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Class Distance Weighted Cross-Entropy loss.

    Penalizes predictions based on their distance from the true class.
    Predictions 1 class away are penalized less than predictions 5 classes away.

    Reference: "SORD: Soft Ordinal Regression for Deep Learning"

    Args:
        logits: [batch_size, num_classes] class logits
        labels: [batch_size] integer class labels
        num_classes: Total number of classes

    Returns:
        loss: Scalar loss value
    """
    batch_size = logits.size(0)

    # Get predicted probabilities
    F.softmax(logits, dim=1)

    # Create distance matrix
    classes = torch.arange(num_classes, device=logits.device)
    labels_expanded = labels.unsqueeze(1)  # [batch_size, 1]

    # Distance of each class from true label
    distances = torch.abs(
        classes - labels_expanded
    ).float()  # [batch_size, num_classes]

    # Weight increases with distance: weight = 1 + distance
    weights = 1.0 + distances

    # Standard CE loss
    log_probs = F.log_softmax(logits, dim=1)
    ce_loss = -log_probs[torch.arange(batch_size), labels]

    # Weight the loss based on how far the predicted distribution is from true class
    # Higher weight for predictions far from true class
    weighted_loss = ce_loss * weights[torch.arange(batch_size), labels]

    return weighted_loss.mean()


def combined_ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    loss_type: str = "coral",
    weights: dict = None,
) -> torch.Tensor:
    """
    Combined loss function that can switch between different ordinal loss types.

    Args:
        logits: Model logits
        labels: Integer class labels
        num_classes: Total number of classes
        loss_type: One of "coral", "soft_labels", "focal", "cdw_ce", "mse"
        weights: Optional dict with loss-specific parameters

    Returns:
        loss: Scalar loss value
    """
    if weights is None:
        weights = {}

    if loss_type == "coral":
        return coral_loss(logits, labels, num_classes)

    elif loss_type == "soft_labels":
        sigma = weights.get("sigma", 1.0)
        return soft_label_cross_entropy(logits, labels, num_classes, sigma=sigma)

    elif loss_type == "focal":
        alpha = weights.get("alpha", 0.25)
        gamma = weights.get("gamma", 2.0)
        return focal_loss(logits, labels, alpha=alpha, gamma=gamma)

    elif loss_type == "cdw_ce":
        return cdw_ce_loss(logits, labels, num_classes)

    elif loss_type == "mse":
        # Standard regression MSE (for baseline comparison)
        # Assumes logits are single regression output
        return F.mse_loss(logits.squeeze(), labels.float())

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Alias for backward compatibility
def get_loss_function(loss_type: str):
    """
    Factory function to get loss function by name.

    Args:
        loss_type: One of "coral", "soft_labels", "focal", "cdw_ce", "mse"

    Returns:
        Loss function
    """
    loss_functions = {
        "coral": coral_loss,
        "soft_labels": soft_label_cross_entropy,
        "focal": focal_loss,
        "cdw_ce": cdw_ce_loss,
    }

    if loss_type not in loss_functions:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Choose from {list(loss_functions.keys())}"
        )

    return loss_functions[loss_type]
