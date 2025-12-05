#!/usr/bin/env python3
"""
Validate BIO tag alignment by inspecting actual examples.
Shows text, tokens, BIO tags, and M2 annotations side-by-side.
"""

import sys
import json
import random
from pathlib import Path

sys.path.append("scripts/training")

from transformers import AutoTokenizer
from parse_m2_annotations import M2Annotation, align_m2_to_subword_tokens


def visualize_example(example, tokenizer):
    """Visualize one example with BIO tags."""

    # Get annotated sentences
    annotated_sents = example.get("annotated_sentences", [])

    if not annotated_sents:
        print("⚠️  No annotations for this example")
        return False

    # Combine first 5 sentences
    combined_text_parts = []
    all_annotations = []

    for sent_data in annotated_sents[:5]:
        sent_text = sent_data.get("text", "")
        combined_text_parts.append(sent_text)

        for ann_data in sent_data.get("annotations", []):
            m2_ann = M2Annotation(
                start_token=ann_data["start_token"],
                end_token=ann_data["end_token"],
                error_type=ann_data.get("error_type", ""),
                correction=ann_data.get("correction", ""),
                required=ann_data.get("required", True),
            )
            all_annotations.append(m2_ann)

    if not all_annotations:
        print("⚠️  No error annotations in first 5 sentences")
        return False

    combined_text = " ".join(combined_text_parts)

    # Get BIO tags
    bio_tags = align_m2_to_subword_tokens(combined_text, all_annotations, tokenizer)
    tokens = tokenizer.tokenize(combined_text)

    # Display
    print("\n" + "=" * 80)
    print("ESSAY INFO:")
    print("=" * 80)
    print(f"CEFR: {example.get('cefr', 'N/A')} (score: {example.get('target', 0):.1f})")
    print(f"Error count: {example.get('error_count', 0)}")
    print(f"Error distribution: {example.get('error_distribution', {})}")

    print("\n" + "=" * 80)
    print("TEXT (first 5 sentences):")
    print("=" * 80)
    print(combined_text[:500] + ("..." if len(combined_text) > 500 else ""))

    print("\n" + "=" * 80)
    print("M2 ANNOTATIONS:")
    print("=" * 80)
    space_tokens = combined_text.split()
    for i, ann in enumerate(all_annotations, 1):
        error_tokens = space_tokens[ann.start_token : ann.end_token]
        print(
            f"{i}. Tokens {ann.start_token}-{ann.end_token}: {' '.join(error_tokens)}"
        )
        print(f"   Type: {ann.error_type}")
        print(f"   Correction: '{ann.correction}'")

    print("\n" + "=" * 80)
    print("SUBWORD TOKENS WITH BIO TAGS:")
    print("=" * 80)

    # Group by BIO tag for easier reading
    error_tokens = []
    correct_tokens = []

    for token, tag in zip(tokens[:50], bio_tags[:50]):  # Show first 50
        marker = "❌" if tag in ["B-ERROR", "I-ERROR"] else "✅"
        token_display = f"{marker} {token:20s} → {tag}"

        if tag in ["B-ERROR", "I-ERROR"]:
            error_tokens.append(token_display)
        else:
            correct_tokens.append(token_display)

    if error_tokens:
        print("\nERROR TOKENS:")
        for t in error_tokens:
            print(f"  {t}")
    else:
        print("\n⚠️  NO ERROR TOKENS DETECTED!")

    print(f"\nCORRECT TOKENS: {len(correct_tokens)} tokens marked as correct")
    if len(correct_tokens) <= 10:
        for t in correct_tokens[:10]:
            print(f"  {t}")

    # Statistics
    error_count = sum(1 for tag in bio_tags if tag in ["B-ERROR", "I-ERROR"])
    print("\n📊 STATISTICS:")
    print(f"   Total tokens: {len(tokens)}")
    print(f"   Error tokens: {error_count} ({error_count / len(tokens) * 100:.1f}%)")
    print(f"   M2 annotations: {len(all_annotations)}")

    # Verdict
    if error_count > 0:
        print("\n✅ LOOKS GOOD: Errors are being detected")
        return True
    else:
        print("\n❌ PROBLEM: No errors detected despite M2 annotations")
        return False


def main():
    print("=" * 80)
    print("BIO TAG VALIDATION")
    print("=" * 80)
    print("\nLoading dataset and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Load dev set
    data_file = Path("data-enhanced/dev-enhanced.jsonl")

    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return

    # Load all examples
    examples = []
    with open(data_file) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Filter to examples with annotations
    examples_with_annotations = [
        e
        for e in examples
        if e.get("annotated_sentences")
        and any(
            sent.get("annotations") for sent in e.get("annotated_sentences", [])[:5]
        )
    ]

    print(f"Examples with annotations: {len(examples_with_annotations)}")

    # Sample 20-30 random examples
    num_to_inspect = min(25, len(examples_with_annotations))
    sampled = random.sample(examples_with_annotations, num_to_inspect)

    print(f"\nInspecting {num_to_inspect} random examples...")
    print("Press Enter to see next example, 'q' to quit\n")

    good_count = 0
    bad_count = 0

    for i, example in enumerate(sampled, 1):
        print(f"\n{'#' * 80}")
        print(f"EXAMPLE {i}/{num_to_inspect}")
        print(f"{'#' * 80}")

        is_good = visualize_example(example, tokenizer)

        if is_good:
            good_count += 1
        else:
            bad_count += 1

        if i < num_to_inspect:
            response = input("\nPress Enter for next, 'q' to quit: ").strip().lower()
            if response == "q":
                break

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Examples inspected: {good_count + bad_count}")
    print(
        f"✅ Good (errors detected): {good_count} ({good_count / (good_count + bad_count) * 100:.0f}%)"
    )
    print(
        f"❌ Bad (no errors detected): {bad_count} ({bad_count / (good_count + bad_count) * 100:.0f}%)"
    )

    if good_count / (good_count + bad_count) >= 0.8:
        print("\n✅ PASS: BIO tagging appears to be working well!")
        print("   Recommend proceeding with quick test training.")
    elif good_count / (good_count + bad_count) >= 0.5:
        print("\n⚠️  MARGINAL: Some examples working, some not.")
        print("   May want to investigate edge cases before training.")
    else:
        print("\n❌ FAIL: Most examples not detecting errors.")
        print("   Need to debug alignment function before training.")


if __name__ == "__main__":
    main()
