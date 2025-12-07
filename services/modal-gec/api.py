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
    import time

    import torch

    _load_model()

    # Smart chunking using Spacy - track positions using span offsets
    # Each item is (chunk_text, chunk_start_char)
    chunks_with_positions: list[tuple[str, int]] = []

    chunk_start_time = time.time()
    if hasattr(_extractor, "annotator") and _extractor.annotator:
        try:
            doc = _extractor.annotator.parse(text)
            chunk_start = -1
            chunk_end = 0

            for sent in doc.sents:
                sent_len = sent.end_char - sent.start_char
                current_chunk_len = chunk_end - chunk_start if chunk_start >= 0 else 0

                # Keep chunks relatively small (<400 chars) to prevent hallucination loops
                if chunk_start < 0:
                    # Start a new chunk
                    chunk_start = sent.start_char
                    chunk_end = sent.end_char
                elif current_chunk_len + sent_len < 400:
                    # Extend current chunk to include this sentence
                    chunk_end = sent.end_char
                else:
                    # Save current chunk and start a new one
                    chunk_text = text[chunk_start:chunk_end]
                    if chunk_text.strip():
                        chunks_with_positions.append((chunk_text, chunk_start))
                    chunk_start = sent.start_char
                    chunk_end = sent.end_char

            # Don't forget the last chunk
            if chunk_start >= 0:
                chunk_text = text[chunk_start:chunk_end]
                if chunk_text.strip():
                    chunks_with_positions.append((chunk_text, chunk_start))

        except Exception as e:
            print(f"WARNING: Smart chunking failed: {e}. Falling back to simple split.")
            # Fallback: split by newlines and track positions using text.find()
            search_start = 0
            for line in text.split("\n"):
                if line.strip():
                    pos = text.find(line, search_start)
                    if pos >= 0:
                        chunks_with_positions.append((line, pos))
                        search_start = pos + len(line)
    else:
        # No spacy: split by newlines and track positions using text.find()
        search_start = 0
        for line in text.split("\n"):
            if line.strip():
                pos = text.find(line, search_start)
                if pos >= 0:
                    chunks_with_positions.append((line, pos))
                    search_start = pos + len(line)

    # Filter valid chunks (already done during chunking)
    valid_chunks = chunks_with_positions
    chunk_time = time.time() - chunk_start_time

    if not valid_chunks:
        print("DEBUG: No valid chunks to process")
        return CorrectionResponse(original=text, corrected=text, edits=[])

    # Log chunk info
    print(
        f"DEBUG: Chunking took {chunk_time:.2f}s - {len(valid_chunks)} chunks from {len(text)} chars"
    )
    for i, (chunk, pos) in enumerate(valid_chunks):
        print(f"  Chunk {i}: pos={pos}, len={len(chunk)}, text='{chunk[:40]}...'")

    # Prepare batched inputs - prefix all chunks with "grammar: "
    input_texts = [f"grammar: {chunk}" for chunk, _ in valid_chunks]

    # Batch tokenization with padding
    import time

    gen_start = time.time()

    with torch.inference_mode():
        inputs = _tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,  # Pad to longest in batch
        ).to(_model.device)

        # Smart max_length: input length + 20% buffer (corrections rarely add much)
        input_len = inputs.input_ids.shape[1]
        max_output_len = min(512, int(input_len * 1.2) + 10)
        print(
            f"DEBUG: Tokenized batch: {inputs.input_ids.shape}, max_output={max_output_len}"
        )

        # Batched generation - optimized for speed
        outputs = _model.generate(
            **inputs,
            max_length=max_output_len,
            num_beams=1,  # Greedy decoding - ~2x faster than beam search
            do_sample=False,  # Deterministic output
        )

    gen_time = time.time() - gen_start
    print(f"DEBUG: Generation took {gen_time:.2f}s for {len(valid_chunks)} chunks")

    # Decode all outputs
    corrected_texts = _tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract edits for each chunk (ERRANT uses spacy to parse and compare)
    extract_start = time.time()
    corrected_chunks = []
    all_edits = []

    for i, (chunk, chunk_start) in enumerate(valid_chunks):
        corrected_chunk_text = corrected_texts[i]
        corrected_chunks.append(corrected_chunk_text)

        # Extract edits for this chunk (ERRANT parses both texts with spacy)
        chunk_edits = _extractor.extract_edits(chunk, corrected_chunk_text)

        # Log chunk corrections
        if chunk_edits:
            print(f"DEBUG: Chunk {i} has {len(chunk_edits)} edits")
            for e in chunk_edits[:3]:  # Show first 3
                print(
                    f"    Edit: pos={e['start']}-{e['end']} '{e['original']}' -> '{e['correction']}'"
                )

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

    extract_time = time.time() - extract_start
    print(
        f"DEBUG: Extraction took {extract_time:.2f}s - {len(all_edits)} total edits found"
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
