"""Logits processing utilities for different model types."""

import numpy as np
import torch
import torch.nn.functional as F


def process_1d_logits(logits_np: np.ndarray, length: int) -> list[float]:
    """Process 1D logits array."""
    if length == 36:
        logits_reshaped = logits_np.reshape(6, 6)
        raw_scores = []
        for dim_logits in logits_reshaped:
            logits_tensor = torch.tensor(dim_logits)
            probs = F.softmax(logits_tensor, dim=0).numpy()
            class_scores_1to5 = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]
            weighted_score = sum(probs[i] * class_scores_1to5[i] for i in range(len(probs)))
            raw_scores.append(float(weighted_score))
        return raw_scores
    elif length == 6:
        logits_tensor = torch.tensor(logits_np)
        max_val = float(torch.max(logits_tensor))
        min_val = float(torch.min(logits_tensor))
        if min_val >= -2 and max_val <= 10 and abs(max_val - min_val) < 5:
            return list(np.clip(logits_np, 1.0, 5.0).tolist())
        else:
            return list(np.clip(logits_np, 1.0, 5.0).tolist())
    else:
        if length > 6:
            return list(np.clip(logits_np[:6], 1.0, 5.0).tolist())
        else:
            return list((np.clip(logits_np, 1.0, 5.0).tolist() * 6)[:6])


def process_2d_logits(logits_np: np.ndarray) -> list[float]:
    """Process 2D logits array."""
    if logits_np.shape[1] == 36:
        logits_reshaped = logits_np[0].reshape(6, 6)
        raw_scores = []
        for dim_logits in logits_reshaped:
            logits_tensor = torch.tensor(dim_logits)
            probs = F.softmax(logits_tensor, dim=0).numpy()
            class_scores_1to5 = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]
            weighted_score = sum(probs[i] * class_scores_1to5[i] for i in range(len(probs)))
            raw_scores.append(float(weighted_score))
        return raw_scores
    elif logits_np.shape[0] == 6 and logits_np.shape[1] == 6:
        raw_scores = []
        for dim_logits in logits_np:
            logits_tensor = torch.tensor(dim_logits)
            probs = F.softmax(logits_tensor, dim=0).numpy()
            class_scores_1to5 = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]
            weighted_score = sum(probs[i] * class_scores_1to5[i] for i in range(len(probs)))
            raw_scores.append(float(weighted_score))
        return raw_scores
    elif logits_np.shape[0] == 1 and logits_np.shape[1] == 6:
        return list(np.clip(logits_np[0], 1.0, 5.0).tolist())
    else:
        flat = logits_np.flatten()
        if len(flat) >= 36:
            logits_reshaped = flat[:36].reshape(6, 6)
            raw_scores = []
            for dim_logits in logits_reshaped:
                logits_tensor = torch.tensor(dim_logits)
                probs = F.softmax(logits_tensor, dim=0).numpy()
                class_scores_1to5 = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]
                weighted_score = sum(probs[i] * class_scores_1to5[i] for i in range(len(probs)))
                raw_scores.append(float(weighted_score))
            return raw_scores
        else:
            clipped = np.clip(flat[:6], 1.0, 5.0)
            return [float(x) for x in clipped.tolist()]


def process_engessay_logits(logits_np: np.ndarray) -> list[float]:
    """Process Engessay model logits into raw scores."""
    if len(logits_np.shape) == 0:
        return [float(logits_np)] * 6
    elif len(logits_np.shape) == 1:
        return process_1d_logits(logits_np, len(logits_np))
    elif len(logits_np.shape) == 2:
        return process_2d_logits(logits_np)
    else:
        flat = logits_np.flatten()
        if len(flat) >= 36:
            logits_reshaped = flat[:36].reshape(6, 6)
            raw_scores = []
            for dim_logits in logits_reshaped:
                logits_tensor = torch.tensor(dim_logits)
                probs = F.softmax(logits_tensor, dim=0).numpy()
                class_scores_1to5 = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]
                weighted_score = sum(probs[i] * class_scores_1to5[i] for i in range(len(probs)))
                raw_scores.append(float(weighted_score))
            return raw_scores
        else:
            clipped = np.clip(flat[:6], 1.0, 5.0)
            return [float(x) for x in clipped.tolist()]


def process_distilbert_logits(logits_np: np.ndarray) -> float:
    """Process DistilBERT model logits into raw score."""
    if len(logits_np.shape) == 0:
        return float(logits_np)
    elif len(logits_np.shape) == 1:
        if len(logits_np) == 1:
            return float(logits_np[0])
        else:
            probs = F.softmax(torch.tensor(logits_np), dim=0).numpy()
            return float(sum(i * probs[i] for i in range(len(probs))))
    else:
        return float(np.mean(logits_np))


def normalize_distilbert_score(raw_score: float) -> float:
    """Normalize DistilBERT score to 0-9 band scale."""
    if raw_score > 10:
        normalized_score = (raw_score / 100.0) * 9.0
    elif raw_score > 6:
        normalized_score = (raw_score / 10.0) * 9.0
    elif raw_score > 1:
        normalized_score = (raw_score / 5.0) * 9.0
    else:
        normalized_score = raw_score * 9.0 if raw_score <= 1.0 else min(9.0, max(0.0, raw_score))

    overall_score = round(normalized_score * 2) / 2
    return max(0.0, min(9.0, overall_score))
