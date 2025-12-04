#!/usr/bin/env python3
"""
Test that the trained corpus model can be loaded and used as an assessor.

This script tests:
1. Model can be loaded from Modal volume
2. Model can score an essay
3. Results are in correct format
"""

import os
import sys
from pathlib import Path

# Add services to path - go up from scripts/training to root, then to services/modal-essay
root_path = Path(__file__).parent.parent.parent
services_path = root_path / "services" / "modal-essay"
sys.path.insert(0, str(services_path))

# Change to services directory for imports
original_dir = os.getcwd()
os.chdir(services_path)

# Import from modal-essay service (relative imports)
# These imports must be after os.chdir, so we ignore E402
import config as modal_config  # type: ignore[import-untyped]  # noqa: E402
import model_loader  # type: ignore[import-untyped]  # noqa: E402
import scoring  # type: ignore[import-untyped]  # noqa: E402
from api import handlers_submission  # type: ignore[import-untyped]  # noqa: E402


def test_model_loading():
    """Test that the corpus-roberta model can be loaded."""
    print("=" * 80)
    print("TESTING MODEL LOADING")
    print("=" * 80)

    model_key = "corpus-roberta"

    if model_key not in modal_config.MODEL_CONFIGS:
        print(f"❌ ERROR: Model key '{model_key}' not found in MODEL_CONFIGS")
        return False, None, None

    config = modal_config.MODEL_CONFIGS[model_key]
    print(f"Model config: {config}")

    try:
        print(f"\nLoading model: {model_key}...")
        model, tokenizer = model_loader.load_model(model_key)
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Tokenizer type: {type(tokenizer)}")
        return True, model, tokenizer
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None


def test_scoring(model, tokenizer):
    """Test that the model can score an essay."""
    print("\n" + "=" * 80)
    print("TESTING ESSAY SCORING")
    print("=" * 80)

    # Test essay
    question_text = "Write about your favorite hobby."
    answer_text = "My favorite hobby is reading books. I enjoy reading because it helps me learn new things and relax. I usually read in the evening before going to bed. Reading makes me feel calm and happy."

    print(f"\nQuestion: {question_text}")
    print(f"Answer: {answer_text[:100]}...")

    try:
        scores = scoring.score_essay(
            question_text=question_text,
            answer_text=answer_text,
            model=model,
            tokenizer=tokenizer,
            model_key="corpus-roberta",
        )

        print("\n✅ Scoring successful!")
        print("\nScores:")
        for dimension, score in scores.items():
            print(f"  {dimension}: {score:.2f}")

        # Validate scores
        overall = scores.get("Overall", scores.get("overall", 0))
        if 2.0 <= overall <= 9.0:
            print(f"\n✅ Overall score in valid range: {overall:.2f}")
        else:
            print(f"\n⚠️  Overall score out of range: {overall:.2f} (expected 2.0-9.0)")

        return True, scores
    except Exception as e:
        print(f"❌ ERROR: Failed to score essay: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_assessor_format(scores):
    """Test that scores can be formatted as AssessorResult."""
    print("\n" + "=" * 80)
    print("TESTING ASSESSOR RESULT FORMAT")
    print("=" * 80)

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

        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to create AssessorResult: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🧪 Testing Corpus Model Integration\n")

    try:
        # Test 1: Load model
        success, model, tokenizer = test_model_loading()
        if not success:
            print("\n❌ Model loading failed. Cannot continue tests.")
            return 1

        # Test 2: Score essay
        success, scores = test_scoring(model, tokenizer)
        if not success:
            print("\n❌ Essay scoring failed.")
            return 1

        # Test 3: Format as assessor result
        success = test_assessor_format(scores)
        if not success:
            print("\n❌ AssessorResult formatting failed.")
            return 1

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe corpus-roberta model is ready to use as an assessor.")
        print("You can now proceed with full training.")
        return 0
    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    sys.exit(main())
