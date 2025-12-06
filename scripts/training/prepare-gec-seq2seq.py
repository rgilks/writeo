#!/usr/bin/env python3
"""
Prepare data for GEC Seq2Seq training.

Converts M2 files (Source + Edits) into Source-Target pairs.
Also applies data augmentation (synthetic errors) to increase dataset size.

Input: M2 files
Output: JSONL files with {"source": "...", "target": "..."}
"""

import json
import random
import sys
from pathlib import Path

# Add current directory to path to import parse_m2_annotations
sys.path.append(str(Path(__file__).parent))
from parse_m2_annotations import parse_m2_file, M2Sentence

# Configuration
OUTPUT_DIR = Path("data/gec-seq2seq")
DATA_DIR = Path("scripts/training/corpus-raw/user-prompt-final-versions")

TRAIN_M2 = DATA_DIR / "en-writeandimprove2024-final-versions-train-sentences.m2"
DEV_M2 = DATA_DIR / "en-writeandimprove2024-final-versions-dev-sentences.m2"
TEST_M2 = (
    DATA_DIR / "en-writeandimprove2024-final-versions-test-sentences.m2"
)  # Likely doesn't exist or is blind


def apply_corrections(sentence: M2Sentence) -> str:
    """
    Apply M2 annotations to source text to create corrected target text.

    Args:
        sentence: M2Sentence object with tokens and annotations

    Returns:
        Corrected sentence string
    """
    if not sentence.annotations:
        return sentence.text

    # Sort annotations by start token descending to avoid index shifting
    # Filter out noop/noop-like annotations just in case
    edits = sorted(
        [a for a in sentence.annotations if a.error_type != "noop"],
        key=lambda x: x.start_token,
        reverse=True,
    )

    tokens = list(sentence.tokens)

    for edit in edits:
        # Check bounds
        if edit.start_token > len(tokens) or edit.end_token > len(tokens):
            continue

        # Replace the span [start:end] with correction tokens
        # Correction string needs to be split into tokens
        correction_tokens = edit.correction.split()

        # Slice replacement
        tokens[edit.start_token : edit.end_token] = correction_tokens

    return " ".join(tokens)


def augment_sentence(sentence: str) -> str:
    """
    Generate a synthetic error version of a correct sentence.

    Simple rule-based augmentations:
    1. Delete random word (Simulate missing word)
    2. Swap adjacent words (Simulate word order error)
    3. Duplicate word (Simulate repetition)
    """
    tokens = sentence.split()
    if len(tokens) < 4:
        return sentence

    aug_type = random.choice(["delete", "swap", "duplicate", "none"])

    if aug_type == "none":
        return sentence

    idx = random.randint(0, len(tokens) - 1)

    if aug_type == "delete":
        # Don't delete if it makes sentence too short
        if len(tokens) > 3:
            tokens.pop(idx)

    elif aug_type == "swap" and idx < len(tokens) - 1:
        tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]

    elif aug_type == "duplicate":
        tokens.insert(idx, tokens[idx])

    return " ".join(tokens)


def process_file(m2_path: Path, output_path: Path, augment: bool = False):
    """Process a single M2 file and save as JSONL."""
    if not m2_path.exists():
        print(f"Warning: {m2_path} not found. Skipping.")
        return

    print(f"Processing {m2_path}...")
    sentences = parse_m2_file(m2_path)

    examples = []

    for sent in sentences:
        source_text = sent.text
        target_text = apply_corrections(sent)

        # Add real example
        examples.append({"source": source_text, "target": target_text, "type": "real"})

        # Data Augmentation (only if sentence was originally correct)
        # If sentence had no errors, source == target. Use it to generate synthetic error.
        if augment and source_text == target_text:
            # Create synthetic error
            corrupt_source = augment_sentence(target_text)
            if corrupt_source != target_text:
                examples.append(
                    {
                        "source": corrupt_source,
                        "target": target_text,
                        "type": "synthetic",
                    }
                )

    # Save to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")


def main():
    random.seed(42)  # Reproducibility

    # Process Train (with augmentation)
    process_file(TRAIN_M2, OUTPUT_DIR / "train.jsonl", augment=True)

    # Process Dev (no augmentation)
    process_file(DEV_M2, OUTPUT_DIR / "dev.jsonl", augment=False)

    # Process Test (no augmentation)
    # Note: Test set usually doesn't have annotations (gold), but if it does, handle it.
    # If M2 has annotations, we can use it for evaluation.
    process_file(TEST_M2, OUTPUT_DIR / "test.jsonl", augment=False)


if __name__ == "__main__":
    main()
