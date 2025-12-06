"""FastAPI API for GEC service."""

import os
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI


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
    edits: List[Edit]


# Global model holder (loaded once on startup)
_model = None
_tokenizer = None
_extractor = None


def _load_model():
    """Load the model on first request (lazy loading)."""
    global _model, _tokenizer, _extractor

    if _model is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from correction import ErrorExtractor

    # Check if fine-tuned model exists
    model_path = "/checkpoints/gec-seq2seq-v1/checkpoint-4398"
    print(f"DEBUG: Checking {model_path}")

    if not os.path.exists(model_path):
        print(f"Specific checkpoint {model_path} not found, checking root...")
        model_path = "/checkpoints/gec-seq2seq-v1"

    if not os.path.exists(model_path) or not os.listdir(model_path):
        print("No fine-tuned model found, using google/flan-t5-base")
        model_name = "google/flan-t5-base"
    else:
        print(f"Loading fine-tuned model from {model_path}")
        model_name = model_path

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda()
    _extractor = ErrorExtractor(use_errant=True)
    print("Model loaded successfully.")


def _correct_text(text: str) -> CorrectionResponse:
    """Correct grammar in the given text using batched inference."""
    import torch

    _load_model()

    # Smart chunking using Spacy - track positions using span offsets
    # Each item is (chunk_text, chunk_start_char)
    chunks_with_positions: list[tuple[str, int]] = []

    if hasattr(_extractor, "annotator") and _extractor.annotator:
        try:
            doc = _extractor.annotator.parse(text)
            current_chunk = ""
            chunk_start = 0

            for sent in doc.sents:
                sent_text = text[sent.start_char : sent.end_char]
                # Keep chunks relatively small (<400 chars) to prevent hallucination loops
                if len(current_chunk) + len(sent_text) < 400:
                    if not current_chunk:
                        chunk_start = sent.start_char
                    current_chunk += sent_text
                else:
                    if current_chunk.strip():
                        chunks_with_positions.append((current_chunk, chunk_start))
                    current_chunk = sent_text
                    chunk_start = sent.start_char

            if current_chunk.strip():
                chunks_with_positions.append((current_chunk, chunk_start))

        except Exception as e:
            print(f"WARNING: Smart chunking failed: {e}. Falling back to simple split.")
            # Fallback: split by newlines and track positions
            pos = 0
            for line in text.split("\n"):
                if line.strip():
                    chunks_with_positions.append((line, pos))
                pos += len(line) + 1  # +1 for newline
    else:
        # No spacy: split by newlines and track positions
        pos = 0
        for line in text.split("\n"):
            if line.strip():
                chunks_with_positions.append((line, pos))
            pos += len(line) + 1  # +1 for newline

    # Filter valid chunks (already done during chunking)
    valid_chunks = chunks_with_positions

    if not valid_chunks:
        return CorrectionResponse(original=text, corrected=text, edits=[])

    # Prepare batched inputs - prefix all chunks with "grammar: "
    input_texts = [f"grammar: {chunk}" for chunk, _ in valid_chunks]

    # Batch tokenization with padding
    with torch.inference_mode():
        inputs = _tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,  # Pad to longest in batch
        ).to(_model.device)

        # Batched generation - process all chunks in one GPU call
        outputs = _model.generate(
            **inputs,
            max_length=512,
            num_beams=2,  # Reduced from 4 for speed (still good quality)
            early_stopping=True,
        )

    # Decode all outputs
    corrected_texts = _tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract edits for each chunk
    corrected_chunks = []
    all_edits = []

    for i, (chunk, chunk_start) in enumerate(valid_chunks):
        corrected_chunk_text = corrected_texts[i]
        corrected_chunks.append(corrected_chunk_text)

        # Extract edits for this chunk
        chunk_edits = _extractor.extract_edits(chunk, corrected_chunk_text)

        # Adjust offsets for global position
        for e in chunk_edits:
            all_edits.append(
                Edit(
                    start=e["start"] + chunk_start,
                    end=e["end"] + chunk_start,
                    original=e["original"],
                    correction=e["correction"],
                    operation=e["operation"],
                    category=e["category"],
                )
            )

    # Reconstruct corrected text - use space join for spacy chunking, newline for fallback
    if (
        hasattr(_extractor, "annotator")
        and _extractor.annotator
        and len(valid_chunks) > 1
    ):
        # We used spacy chunking - join with space
        final_corrected_text = " ".join(corrected_chunks)
    else:
        final_corrected_text = "\n".join(corrected_chunks)

    return CorrectionResponse(
        original=text, corrected=final_corrected_text, edits=all_edits
    )


def create_fastapi_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Writeo GEC Service",
        description="Grammatical Error Correction using Seq2Seq models",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "ok", "model": "gec-seq2seq"}

    @app.post("/gec_endpoint", response_model=CorrectionResponse)
    def gec_endpoint(request: CorrectionRequest):
        """Correct grammar in the given text."""
        return _correct_text(request.text)

    return app
