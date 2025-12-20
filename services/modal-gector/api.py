"""FastAPI API for GECToR service."""

import os
import time

from fastapi import FastAPI
from pydantic import BaseModel


class CorrectionRequest(BaseModel):
    text: str


class Edit(BaseModel):
    start: int
    end: int
    original: str
    correction: str
    operation: str  # insert, replace, delete
    category: str  # grammar, vocabulary, mechanics, fluency


class CorrectionResponse(BaseModel):
    original: str
    corrected: str
    edits: list[Edit]


# Global model holder (loaded once on startup)
_model = None
_tokenizer = None
_encode = None
_decode = None


def _load_model():
    """Load the GECToR model on first request (lazy loading)."""
    global _model, _tokenizer, _encode, _decode

    if _model is not None:
        return

    import torch
    from gector.modeling import GECToR
    from gector.predict import load_verb_dict
    from transformers import AutoTokenizer

    model_id = "gotutiyan/gector-roberta-base-5k"
    print(f"Loading GECToR model: {model_id}")

    _model = GECToR.from_pretrained(model_id)
    if torch.cuda.is_available():
        _model.cuda()
        print("Model loaded on GPU")
    else:
        print("WARNING: Running on CPU")

    _tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load verb dictionary for morphological transforms
    # Try multiple paths (local dev vs container)
    possible_paths = [
        "/app/data/verb-form-vocab.txt",
        os.path.join(os.path.dirname(__file__), "data", "verb-form-vocab.txt"),
    ]

    verb_dict_path = None
    for path in possible_paths:
        if os.path.exists(path):
            verb_dict_path = path
            break

    if verb_dict_path:
        _encode, _decode = load_verb_dict(verb_dict_path)
        print(f"Loaded verb dictionary from {verb_dict_path}")
    else:
        # Use empty dict as fallback
        print("WARNING: verb-form-vocab.txt not found, some transforms may fail")
        _encode, _decode = {}, {}

    print("GECToR model loaded successfully.")


def _extract_edits(original: str, corrected: str) -> list[dict]:
    """Extract edits by comparing original and corrected text.

    Uses difflib.SequenceMatcher for proper word-level alignment.
    """
    import difflib
    import re

    edits = []

    # Tokenize with position tracking
    def tokenize_with_positions(text: str) -> list[tuple[str, int, int]]:
        """Return list of (word, start, end) tuples."""
        tokens = []
        for match in re.finditer(r"\S+", text):
            tokens.append((match.group(), match.start(), match.end()))
        return tokens

    orig_tokens = tokenize_with_positions(original)
    corr_tokens = tokenize_with_positions(corrected)

    orig_words = [t[0] for t in orig_tokens]
    corr_words = [t[0] for t in corr_tokens]

    # Use SequenceMatcher for alignment
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            # Replacement: one or more words changed
            for k in range(i2 - i1):
                if k < j2 - j1:
                    # 1:1 replacement
                    orig_token = orig_tokens[i1 + k]
                    corr_word = corr_words[j1 + k]
                    edits.append(
                        {
                            "start": orig_token[1],
                            "end": orig_token[2],
                            "original": orig_token[0],
                            "correction": corr_word,
                            "operation": "replace",
                            "category": "grammar",
                        }
                    )
            # Handle extra deletions if orig has more words
            for k in range(j2 - j1, i2 - i1):
                orig_token = orig_tokens[i1 + k]
                edits.append(
                    {
                        "start": orig_token[1],
                        "end": orig_token[2],
                        "original": orig_token[0],
                        "correction": "",
                        "operation": "delete",
                        "category": "grammar",
                    }
                )
            # Handle extra insertions if corr has more words
            for k in range(i2 - i1, j2 - j1):
                # Insert after last original word in this range
                insert_pos = orig_tokens[i2 - 1][2] if i2 > 0 else 0
                edits.append(
                    {
                        "start": insert_pos,
                        "end": insert_pos,
                        "original": "",
                        "correction": corr_words[j1 + k],
                        "operation": "insert",
                        "category": "grammar",
                    }
                )
        elif tag == "delete":
            # Deletion: words in original not in corrected
            for k in range(i1, i2):
                orig_token = orig_tokens[k]
                edits.append(
                    {
                        "start": orig_token[1],
                        "end": orig_token[2],
                        "original": orig_token[0],
                        "correction": "",
                        "operation": "delete",
                        "category": "grammar",
                    }
                )
        elif tag == "insert":
            # Insertion: words in corrected not in original
            insert_pos = orig_tokens[i1 - 1][2] if i1 > 0 else 0
            for k in range(j1, j2):
                edits.append(
                    {
                        "start": insert_pos,
                        "end": insert_pos,
                        "original": "",
                        "correction": corr_words[k],
                        "operation": "insert",
                        "category": "grammar",
                    }
                )

    return edits


def _correct_text(text: str) -> CorrectionResponse:
    """Correct grammar using GECToR model."""
    from gector.predict import predict

    _load_model()

    # Split into sentences for better handling
    sentences = text.replace("\n", " ").split(". ")
    sentences = [
        s.strip() + "." if not s.endswith(".") else s.strip() for s in sentences if s.strip()
    ]

    if not sentences:
        return CorrectionResponse(original=text, corrected=text, edits=[])

    print(f"DEBUG: Processing {len(sentences)} sentences")
    start_time = time.time()

    # Run GECToR prediction
    corrected_sentences = predict(
        _model,
        _tokenizer,
        sentences,
        _encode,
        _decode,
        keep_confidence=0.0,
        min_error_prob=0.0,
        n_iteration=5,  # Iterative correction
        batch_size=min(len(sentences), 8),
    )

    inference_time = time.time() - start_time
    print(f"DEBUG: GECToR inference took {inference_time:.2f}s")

    # Reconstruct corrected text
    corrected_text = " ".join(corrected_sentences)

    # Extract edits
    all_edits = []
    search_pos = 0

    for orig_sent, corr_sent in zip(sentences, corrected_sentences, strict=False):
        if orig_sent != corr_sent:
            # Find sentence position in original text
            sent_start = text.find(orig_sent.rstrip("."), search_pos)
            if sent_start == -1:
                sent_start = search_pos

            # Extract edits for this sentence
            sent_edits = _extract_edits(orig_sent, corr_sent)

            # Adjust positions to global offset
            for e in sent_edits:
                all_edits.append(
                    Edit(
                        start=e["start"] + sent_start,
                        end=e["end"] + sent_start,
                        original=e["original"],
                        correction=e["correction"],
                        operation=e["operation"],
                        category=e["category"],
                    )
                )

            search_pos = sent_start + len(orig_sent)

    print(f"DEBUG: Found {len(all_edits)} edits")

    return CorrectionResponse(original=text, corrected=corrected_text, edits=all_edits)


def create_fastapi_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Writeo GECToR Service",
        description="Fast Grammatical Error Correction using GECToR (Tag, Not Rewrite)",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "ok", "model": "gector-roberta-base-5k"}

    @app.post("/gector_endpoint", response_model=CorrectionResponse)
    def gector_endpoint(request: CorrectionRequest):
        """Correct grammar in the given text using GECToR."""
        return _correct_text(request.text)

    return app
