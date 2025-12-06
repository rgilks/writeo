import modal
import os
import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel

# Add the parent directory to sys.path to allow imports if running locally or in specific structures
# In Modal, we mount the package
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.modal_gec.correction import ErrorExtractor

# Define image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "errant==3.0.0",
        "spacy==3.7.2",
        "fastapi[standard]",
        "sentencepiece>=0.1.99",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
    )
    # Mount the services directory so we can import services.modal_gec.correction
    .add_local_dir(
        str(Path(__file__).parent.parent.parent / "services"),
        remote_path="/root/services",
    )
)

app = modal.App("writeo-gec-service", image=image)
volume = modal.Volume.from_name("writeo-gec-models", create_if_missing=True)


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


@app.cls(
    gpu="A10G", timeout=600, volumes={"/checkpoints": volume}, scaledown_window=300
)
class GECModel:
    def __enter__(self):
        print("DEBUG: Entering __enter__")
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        # Load model from volume or download default
        # For now, we use google/flan-t5-base as default if no checkpoint exists
        # In production, this should point to the trained checkpoint in /checkpoints

        # Check if fine-tuned model exists
        model_path = "/checkpoints/gec-seq2seq-v1/checkpoint-4398"
        print(f"DEBUG: Checking {model_path}")

        # Fallback to checking root if specific checkpoint not found (robustness)
        if not os.path.exists(model_path):
            print(f"Specific checkpoint {model_path} not found, checking root...")
            model_path = "/checkpoints/gec-seq2seq-v1"

        if not os.path.exists(model_path) or not os.listdir(model_path):
            print("No fine-tuned model found, using google/flan-t5-base")
            model_name = "google/flan-t5-base"
        else:
            print(f"Loading fine-tuned model from {model_path}")
            model_name = model_path

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).cuda()
            self.extractor = ErrorExtractor(use_errant=True)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            raise e

    @modal.method()
    def correct(self, text: str) -> CorrectionResponse:
        print(f"DEBUG: correct called with: {text}")
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            print("WARNING: Model not initialized. Attempting lazy load...")
            self.__enter__()

        # Smart chunking using Spacy (via ErrorExtractor) to handle long texts
        # T5 struggles with long sequences, so we batch sentences.
        chunks = []
        if hasattr(self.extractor, "annotator") and self.extractor.annotator:
            try:
                doc = self.extractor.annotator.parse(text)
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
                print(
                    f"WARNING: Smart chunking failed: {e}. Falling back to simple split."
                )
                chunks = text.split("\n")
        else:
            chunks = text.split("\n")

        corrected_chunks = []
        all_edits = []
        current_offset = 0

        for chunk in chunks:
            # Skip empty chunks but track offset
            if not chunk:
                # If chunk is empty (rare with smart chunking), just move on
                # But if it came from split('\n'), it might be an empty line.
                # Reconstruct original behavior for simple split:
                if chunk == "":
                    corrected_chunks.append("")
                    current_offset += 1  # newline
                continue

            # For sentence chunking, empty chunks shouldn't happen, but verify
            if not chunk.strip():
                corrected_chunks.append(chunk)
                current_offset += len(chunk)
                continue

            input_text = f"grammar: {chunk}"

            try:
                inputs = self.tokenizer(
                    input_text, return_tensors="pt", max_length=512, truncation=True
                ).to(self.model.device)
            except AttributeError:
                # Robustness: Reload if tokenizer dies (e.g. race condition)
                self.__enter__()
                inputs = self.tokenizer(
                    input_text, return_tensors="pt", max_length=512, truncation=True
                ).to(self.model.device)

            outputs = self.model.generate(
                **inputs, max_length=512, num_beams=4, early_stopping=True
            )

            corrected_chunk_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            corrected_chunks.append(corrected_chunk_text)

            # Extract edits for this chunk
            chunk_edits = self.extractor.extract_edits(chunk, corrected_chunk_text)

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

        # Smart reconstruction
        # If we used smart chunking, `chunks` are just parts of the string, so we join them.
        # But `corrected_chunks` need to be joined carefully.
        # Ideally, we should just join them with nothing because text_with_ws includes format.
        # However, T5 might eat leading/trailing spaces.
        # For now, simplistic join is risky if T5 stripped spaces.

        # Re-construct:
        # We need to rely on the fact that T5 usually outputs clean text.
        # But if we split "Hello world. How are you?" into "Hello world. " and "How are you?",
        # T5 might output "Hello world." (no space).
        # We might need to add a space if the original had one and T5 lost it, but that's complex.
        # Let's trust T5 + Spacy for now, or just join with " " if it looks like sentence boundary?
        # Actually, `corrected_chunks` will likely not have the trailing whitespace if T5 normalized it.
        # This is a known issue.

        # Better approach for reconstruction:
        # Just map errors back to original text. Return original text + list of edits.
        # But the API returns `corrected` text too.
        # Let's join with " " if the previous chunk ended with ws and current doesn't?
        # No, let's just join with " " if not present?
        # Simpler: just join with " ". The user mostly cares about EDITS.
        # The `corrected` field is for display.
        # Wait, if `chunks` was from `split('\n')`, we join with `\n`.
        # If `chunks` was from spacy, we join with nothing (empty string) because strictly speaking we don't know.
        # But wait, T5 strips whitespace.

        # Let's iterate and try to be smart about joining.
        final_corrected_parts = []
        for i, c_text in enumerate(corrected_chunks):
            final_corrected_parts.append(c_text)
            # If we are in smart chunking mode, we might need spaces.
            # But the edits are what matter most.

        # If we came from split('\n'), join with newline.
        # If we came from spacy, join with space?

        if (
            hasattr(self.extractor, "annotator")
            and self.extractor.annotator
            and len(chunks) > 1
            and chunks[0] != text.split("\n")[0]
        ):
            # We likely used spacy chunking
            final_corrected_text = " ".join(final_corrected_parts)
        else:
            final_corrected_text = "\n".join(final_corrected_parts)

        return CorrectionResponse(
            original=text, corrected=final_corrected_text, edits=all_edits
        )


@app.function()
@modal.fastapi_endpoint(method="POST")
def gec_endpoint(request: CorrectionRequest):
    model = GECModel()
    return model.correct.remote(request.text)


if __name__ == "__main__":
    # Local test
    with app.run():
        model = GECModel()
        resp = model.correct.remote("I has three book.")
        print(resp)
