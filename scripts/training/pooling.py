"""Custom attention pooling layer for metric-specific focus."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling over sequence tokens.

    Instead of using just [CLS] token, learns to weight all tokens
    based on their relevance to essay scoring. This allows the model
    to focus on grammatically complex sentences, vocabulary usage, etc.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-weighted pooling.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Compute attention scores
        scores = self.query(hidden_states).squeeze(-1)  # [batch, seq_len]

        # Mask padding tokens
        scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))

        # Softmax to get attention weights
        weights = F.softmax(scores, dim=1)  # [batch, seq_len]
        weights = self.dropout(weights)

        # Weighted sum of hidden states
        pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden]

        return pooled


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head version of attention pooling for richer representations.

    Uses multiple attention heads, each potentially focusing on different
    aspects of essay quality (grammar, vocabulary, coherence, etc.).
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            "hidden_size must be divisible by num_heads"
        )

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, num_heads)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-head attention-weighted pooling.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute attention scores for each head
        scores = self.query(hidden_states)  # [batch, seq_len, num_heads]

        # Mask padding
        mask = attention_mask.unsqueeze(-1).expand_as(
            scores
        )  # [batch, seq_len, num_heads]
        scores = scores.masked_fill(~mask.bool(), float("-inf"))

        # Softmax per head
        weights = F.softmax(scores, dim=1)  # [batch, seq_len, num_heads]
        weights = self.dropout(weights)

        # Project values
        values = self.value_proj(hidden_states)  # [batch, seq_len, hidden]

        # Reshape for multi-head processing
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Weighted sum per head
        weights_expanded = weights.unsqueeze(-1)  # [batch, seq_len, num_heads, 1]
        pooled = (values * weights_expanded).sum(dim=1)  # [batch, num_heads, head_dim]

        # Concatenate heads
        pooled = pooled.view(batch_size, -1)  # [batch, hidden]

        # Output projection
        return self.output_proj(pooled)
