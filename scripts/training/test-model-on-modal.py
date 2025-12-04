#!/usr/bin/env python3
"""
Test that the trained corpus model can be loaded and used as an assessor on Modal.

This Modal function tests:
1. Model can be loaded from Modal volume
2. Model can score an essay
3. Results are in correct format
"""

import sys
from pathlib import Path

import modal

# Add services to path
root_path = Path(__file__).parent.parent.parent
services_path = root_path / "services" / "modal-essay"
sys.path.insert(0, str(services_path))

# Modal setup
app = modal.App("writeo-test-model")

# Use the same image as the main essay service
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers>=4.40.0",
        "torch==2.2.0",
        "numpy>=1.24.0,<2.0",
    )
    .add_local_dir(str(services_path), remote_path="/services/modal-essay")
)

# Modal volume for model storage
volume = modal.Volume.from_name("writeo-models", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu="T4",  # Use T4 for testing
    timeout=300,  # 5 minutes
)
def test_corpus_model():
    """Test the corpus-roberta model on Modal."""
    import os
    import sys

    # Add services to path
    sys.path.insert(0, "/services/modal-essay")
    os.chdir("/services/modal-essay")

    import model_loader  # type: ignore[import-untyped]
    import scoring  # type: ignore[import-untyped]
    from api import handlers_submission  # type: ignore[import-untyped]

    print("=" * 80)
    print("TESTING CORPUS MODEL ON MODAL")
    print("=" * 80)

    model_key = "corpus-roberta"

    # Test 1: Load model
    print(f"\n1. Loading model: {model_key}...")
    try:
        model, tokenizer = model_loader.load_model(model_key)
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Tokenizer type: {type(tokenizer)}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": f"Model loading failed: {e}"}

    # Test 2: Score essay
    print("\n2. Testing essay scoring...")
    question_text = "Write about your favorite hobby."
    answer_text = "My favorite hobby is reading books. I enjoy reading because it helps me learn new things and relax. I usually read in the evening before going to bed. Reading makes me feel calm and happy."

    try:
        scores = scoring.score_essay(
            question_text=question_text,
            answer_text=answer_text,
            model=model,
            tokenizer=tokenizer,
            model_key=model_key,
        )

        print("✅ Scoring successful!")
        print("\nScores:")
        for dimension, score in scores.items():
            print(f"  {dimension}: {score:.2f}")

        # Validate scores
        overall = scores.get("Overall", scores.get("overall", 0))
        if 2.0 <= overall <= 9.0:
            print(f"\n✅ Overall score in valid range: {overall:.2f}")
        else:
            print(f"\n⚠️  Overall score out of range: {overall:.2f} (expected 2.0-9.0)")
    except Exception as e:
        print(f"❌ ERROR: Failed to score essay: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": f"Scoring failed: {e}"}

    # Test 3: Format as assessor result
    print("\n3. Testing AssessorResult format...")
    try:
        assessor_result = handlers_submission.create_assessor_result(
            scores=scores,
            model_name="corpus-trained-roberta",
            assessor_id="T-AES-CORPUS",
        )

        print("✅ AssessorResult created successfully!")
        print("\nAssessorResult:")
        print(f"  ID: {assessor_result.id}")
        print(f"  Name: {assessor_result.name}")
        print(f"  Type: {assessor_result.type}")
        print(f"  Overall: {assessor_result.overall}")
        print(f"  Label: {assessor_result.label}")
        print("  Dimensions:")
        for dim, score in assessor_result.dimensions.items():
            print(f"    {dim}: {score:.2f}")
    except Exception as e:
        print(f"❌ ERROR: Failed to create AssessorResult: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": f"AssessorResult creation failed: {e}"}

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe corpus-roberta model is ready to use as an assessor.")

    return {
        "success": True,
        "scores": scores,
        "overall": overall,
        "assessor_id": assessor_result.id,
    }


@app.local_entrypoint()
def main():
    """Run the test."""
    result = test_corpus_model.remote()
    print(f"\nTest result: {result}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()
