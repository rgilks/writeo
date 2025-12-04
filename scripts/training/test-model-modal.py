#!/usr/bin/env python3
"""
Quick test to verify the corpus model can be loaded and used on Modal.

Run: modal run scripts/training/test-model-modal.py
"""

import modal

app = modal.App("writeo-test-model")

# Use same image as essay service
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers>=4.40.0",
        "torch==2.2.0",
        "numpy>=1.24.0,<2.0",
        "fastapi==0.104.1",
        "pydantic==2.5.0",
    )
    .add_local_dir("services/modal-essay", remote_path="/app")
)

volume = modal.Volume.from_name("writeo-models", create_if_missing=True)


@app.function(image=image, volumes={"/vol": volume}, gpu="T4", timeout=300)
def test_model():
    """Test loading and using the corpus model."""
    import sys
    import os

    sys.path.insert(0, "/app")
    os.chdir("/app")

    from config import MODEL_CONFIGS  # type: ignore[import-untyped]
    from model_loader import load_model  # type: ignore[import-untyped]
    from scoring import score_essay  # type: ignore[import-untyped]
    from api.handlers_submission import create_assessor_result  # type: ignore[import-untyped]

    model_key = "corpus-roberta"
    print(f"Testing model: {model_key}")
    print(f"Config: {MODEL_CONFIGS[model_key]}")

    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load_model(model_key)
    print("✅ Model loaded!")

    # Score essay
    print("\n2. Scoring essay...")
    scores = score_essay(
        question_text="Write about your favorite hobby.",
        answer_text="My favorite hobby is reading books. I enjoy reading because it helps me learn new things.",
        model=model,
        tokenizer=tokenizer,
        model_key=model_key,
    )
    print(f"✅ Scores: {scores}")

    # Create assessor result
    print("\n3. Creating AssessorResult...")
    result = create_assessor_result(
        scores=scores,
        model_name="corpus-trained-roberta",
        assessor_id="T-AES-CORPUS",
    )
    print(f"✅ AssessorResult: {result.id} - {result.overall:.2f} ({result.label})")

    print("\n✅ All tests passed! Model is ready to use.")
    return {"success": True, "overall": result.overall, "label": result.label}


if __name__ == "__main__":
    with app.run():
        result = test_model.remote()
        print(f"\nResult: {result}")
