"""Utility functions."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    TokenizerType = PreTrainedTokenizer
else:
    TokenizerType = Any


def chunk_text(text: str, tokenizer: TokenizerType, max_tokens: int, overlap: int) -> list[str]:
    """Split text into chunks with overlap.

    Args:
        text: The text to chunk.
        tokenizer: Tokenizer to use for encoding/decoding.
        max_tokens: Maximum number of tokens per chunk.
        overlap: Number of tokens to overlap between chunks.

    Returns:
        List of text chunks.

    Raises:
        ValueError: If overlap >= max_tokens (would cause infinite loop).
    """
    if overlap >= max_tokens:
        raise ValueError(f"Overlap ({overlap}) must be less than max_tokens ({max_tokens})")

    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_str)

        if end >= len(tokens):
            break

        start += max_tokens - overlap

    return chunks
