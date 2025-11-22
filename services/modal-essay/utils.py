"""Utility functions."""

from typing import List


def chunk_text(text: str, tokenizer, max_tokens: int, overlap: int) -> List[str]:
    """Split text into chunks with overlap."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
        
        start += max_tokens - overlap
    
    return chunks

