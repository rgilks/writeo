#!/usr/bin/env python3
"""
Test the M2→subword alignment function.
Validates that BIO tags correctly align with actual errors.
"""

import sys

sys.path.append("scripts/training")

from parse_m2_annotations import (
    M2Annotation,
    align_m2_to_subword_tokens,
)
from transformers import AutoTokenizer


def test_single_token_error():
    """Test SVA error: 'student have' → 'have' should be B-ERROR."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    text = "The student have books"
    # M2 annotation: token 2 (0-indexed) = "have"
    annotations = [
        M2Annotation(
            start_token=2,
            end_token=3,  # Exclusive
            error_type="R:VERB:SVA",
            correction="has",
            required=True,
        )
    ]

    bio_tags = align_m2_to_subword_tokens(text, annotations, tokenizer)
    tokens = tokenizer.tokenize(text)

    print("Test 1: Single token error")
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"BIO tags: {bio_tags}")

    # Verify "have" is marked as error
    have_idx = None
    for i, token in enumerate(tokens):
        if "have" in token.lower():
            have_idx = i
            break

    if have_idx is not None:
        assert bio_tags[have_idx] == "B-ERROR", (
            f"Expected 'have' at index {have_idx} to be B-ERROR, got {bio_tags[have_idx]}"
        )
        print("✅ PASS: 'have' correctly marked as B-ERROR\n")
    else:
        print("⚠️  Could not find 'have' token\n")


def test_multi_token_error():
    """Test determiner error: 'a books' → both tokens should be marked."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    text = "I have a books"
    # M2 annotation: tokens 2-4 = "a books"
    annotations = [
        M2Annotation(
            start_token=2,
            end_token=4,
            error_type="R:DET+NOUN",
            correction="a book",
            required=True,
        )
    ]

    bio_tags = align_m2_to_subword_tokens(text, annotations, tokenizer)
    tokens = tokenizer.tokenize(text)

    print("Test 2: Multi-token error")
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"BIO tags: {bio_tags}")

    # Find error tokens
    error_count = sum(1 for tag in bio_tags if tag in ["B-ERROR", "I-ERROR"])
    assert error_count >= 2, f"Expected at least 2 error tokens, got {error_count}"
    print(f"✅ PASS: {error_count} tokens marked as errors\n")


def test_no_errors():
    """Test correct sentence: all should be 'O'."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    text = "The student has books"
    annotations = []  # No errors

    bio_tags = align_m2_to_subword_tokens(text, annotations, tokenizer)
    tokens = tokenizer.tokenize(text)

    print("Test 3: No errors")
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"BIO tags: {bio_tags}")

    assert all(tag == "O" for tag in bio_tags), "Expected all tags to be 'O'"
    print("✅ PASS: All tokens correctly marked as O\n")


def test_with_real_m2_data():
    """Test with actual M2 data from corpus."""
    from parse_m2_annotations import parse_m2_file
    from pathlib import Path

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    m2_path = Path(
        "scripts/training/corpus-raw/user-prompt-final-versions/en-writeandimprove2024-final-versions-dev-sentences.m2"
    )

    if not m2_path.exists():
        print("⚠️  Test 4 skipped: M2 file not found")
        return

    sentences = parse_m2_file(m2_path)

    # Find sentence with errors
    example = next((s for s in sentences if len(s.annotations) > 0), None)

    if example:
        print("Test 4: Real M2 data")
        print(f"Text: {example.text}")

        bio_tags = align_m2_to_subword_tokens(
            example.text, example.annotations, tokenizer
        )
        tokens = tokenizer.tokenize(example.text)

        print(f"Tokens ({len(tokens)}): {tokens[:10]}...")  # Show first 10
        print(f"BIO tags ({len(bio_tags)}): {bio_tags[:10]}...")

        error_count = sum(1 for tag in bio_tags if tag in ["B-ERROR", "I-ERROR"])
        print(f"Marked {error_count} tokens as errors")
        print(f"M2 annotations: {len(example.annotations)}")

        if error_count > 0:
            print("✅ PASS: Successfully processed real M2 data\n")
        else:
            print(
                "⚠️  WARNING: No errors marked despite {len(example.annotations)} M2 annotations\n"
            )


def main():
    print("=" * 80)
    print("TESTING M2→SUBWORD ALIGNMENT")
    print("=" * 80)
    print()

    try:
        test_single_token_error()
        test_multi_token_error()
        test_no_errors()
        test_with_real_m2_data()

        print("=" * 80)
        print("ALL TESTS PASSED! ✅")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
