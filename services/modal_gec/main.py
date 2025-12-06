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

        input_text = f"grammar: {text}"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs, max_length=512, num_beams=4, early_stopping=True
        )

        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract edits
        edits_data = self.extractor.extract_edits(text, corrected_text)

        # Convert to Pydantic models
        edits_response = []
        for e in edits_data:
            edits_response.append(
                Edit(
                    start=e["start"],
                    end=e["end"],
                    original=e["original"],
                    correction=e["correction"],
                    type=e["type"],
                )
            )

        return CorrectionResponse(
            original=text, corrected=corrected_text, edits=edits_response
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
