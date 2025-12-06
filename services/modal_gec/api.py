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
    type: str  # grammar, vocabulary, mechanics, fluency


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
    """Correct grammar in the given text."""
    _load_model()

    # Smart chunking using Spacy (via ErrorExtractor) to handle long texts
    chunks = []
    if hasattr(_extractor, "annotator") and _extractor.annotator:
        try:
            doc = _extractor.annotator.parse(text)
            sentences = [sent.text_with_ws for sent in doc.sents]
            current_chunk = ""
            for sent in sentences:
                # Keep chunks relatively small (<400 chars) to prevent hallucination loops
                if len(current_chunk) + len(sent) < 400:
                    current_chunk += sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sent
            if current_chunk:
                chunks.append(current_chunk)
        except Exception as e:
            print(f"WARNING: Smart chunking failed: {e}. Falling back to simple split.")
            chunks = text.split("\n")
    else:
        chunks = text.split("\n")

    corrected_chunks = []
    all_edits = []
    current_offset = 0

    for chunk in chunks:
        # Skip empty chunks but track offset
        if not chunk:
            if chunk == "":
                corrected_chunks.append("")
                current_offset += 1  # newline
            continue

        if not chunk.strip():
            corrected_chunks.append(chunk)
            current_offset += len(chunk)
            continue

        input_text = f"grammar: {chunk}"

        inputs = _tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        ).to(_model.device)

        outputs = _model.generate(
            **inputs, max_length=512, num_beams=4, early_stopping=True
        )

        corrected_chunk_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_chunks.append(corrected_chunk_text)

        # Extract edits for this chunk
        chunk_edits = _extractor.extract_edits(chunk, corrected_chunk_text)

        # Adjust offsets for global position
        for e in chunk_edits:
            all_edits.append(
                Edit(
                    start=e["start"] + current_offset,
                    end=e["end"] + current_offset,
                    original=e["original"],
                    correction=e["correction"],
                    type=e["type"],
                )
            )

        # Update offset
        current_offset += len(chunk)

    # Reconstruct corrected text
    if (
        hasattr(_extractor, "annotator")
        and _extractor.annotator
        and len(chunks) > 1
        and chunks[0] != text.split("\n")[0]
    ):
        # We likely used spacy chunking
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
