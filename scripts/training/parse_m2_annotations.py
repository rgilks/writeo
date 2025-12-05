#!/usr/bin/env python3
"""
Parse M2 format error annotations from Write & Improve corpus.

M2 format example:
S There are lots of important things in our life.
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0

S However, in most cases, they depend on the priorities we have.
A 12 12|||M:DET|||the|||REQUIRED|||-NONE-|||0
  ^^  ^^ ^^error^^ ^^correction^^
  start end type
"""

import re
from pathlib import Path
from typing import Any
from dataclasses import dataclass


@dataclass
class M2Annotation:
    """Single M2 error annotation."""

    start_token: int
    end_token: int
    error_type: str
    correction: str
    required: bool


@dataclass
class M2Sentence:
    """Sentence with annotations."""

    text: str
    tokens: list[str]
    annotations: list[M2Annotation]


# Error type mapping to simplified categories
ERROR_CATEGORY_MAP = {
    # Grammar errors
    "R:VERB": "grammar",
    "R:VERB:SVA": "grammar",
    "R:VERB:TENSE": "grammar",
    "R:VERB:FORM": "grammar",
    "R:NOUN": "grammar",
    "R:NOUN:NUM": "grammar",
    "R:DET": "grammar",
    "M:DET": "grammar",
    "R:PRON": "grammar",
    "R:ADJ": "grammar",
    "R:ADV": "grammar",
    "R:CONJ": "grammar",
    # Vocabulary/word choice
    "R:WO": "vocabulary",  # Word order
    "R:OTHER": "vocabulary",
    # Prepositions
    "R:PREP": "grammar",
    "M:PREP": "grammar",
    "U:PREP": "grammar",
    # Mechanics
    "R:ORTH": "mechanics",  # Orthography/spelling
    "R:PUNCT": "mechanics",
    # Fluency/style
    "R:PART": "fluency",
    "U:VERB": "fluency",
    # Additions/deletions
    "U:": "fluency",  # Unnecessary word
    "M:": "grammar",  # Missing word
}


def parse_m2_file(m2_path: Path) -> list[M2Sentence]:
    """
    Parse M2 format file into structured annotations.

    Format:
        S <sentence text>
        A <start> <end>|||<error_type>|||<correction>|||REQUIRED|||-NONE-|||0
        A ...

        S <next sentence>
        ...
    """
    sentences = []
    current_sentence = None
    current_annotations = []

    with open(m2_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("S "):
                # New sentence
                if current_sentence:
                    # Save previous sentence
                    sentences.append(
                        M2Sentence(
                            text=current_sentence,
                            tokens=current_sentence.split(),
                            annotations=current_annotations,
                        )
                    )

                current_sentence = line[2:]  # Remove "S "
                current_annotations = []

            elif line.startswith("A "):
                # Annotation line
                parts = line[2:].split("|||")
                if len(parts) >= 5:
                    start_end = parts[0].split()
                    if len(start_end) == 2:
                        start_token = int(start_end[0])
                        end_token = int(start_end[1])
                        error_type = parts[1]
                        correction = parts[2]
                        required = parts[3] == "REQUIRED"

                        # Skip noop annotations
                        if error_type != "noop":
                            current_annotations.append(
                                M2Annotation(
                                    start_token=start_token,
                                    end_token=end_token,
                                    error_type=error_type,
                                    correction=correction,
                                    required=required,
                                )
                            )

        # Don't forget the last sentence
        if current_sentence:
            sentences.append(
                M2Sentence(
                    text=current_sentence,
                    tokens=current_sentence.split(),
                    annotations=current_annotations,
                )
            )

    return sentences


def map_error_to_category(error_type: str) -> str:
    """Map specific error type to general category."""
    # Direct match
    if error_type in ERROR_CATEGORY_MAP:
        return ERROR_CATEGORY_MAP[error_type]

    # Prefix match (e.g., "R:VERB:INFL" -> "grammar")
    for prefix, category in ERROR_CATEGORY_MAP.items():
        if error_type.startswith(prefix):
            return category

    # Default
    return "other"


def create_bio_tags(sentence: M2Sentence) -> list[str]:
    """
    Create BIO tags for token-level error detection.

    B-ERROR: Beginning of error span
    I-ERROR: Inside error span
    O: Outside (correct)
    """
    num_tokens = len(sentence.tokens)
    tags = ["O"] * num_tokens

    for ann in sentence.annotations:
        if 0 <= ann.start_token < num_tokens:
            tags[ann.start_token] = "B-ERROR"

            for i in range(ann.start_token + 1, min(ann.end_token, num_tokens)):
                tags[i] = "I-ERROR"

    return tags


def extract_error_statistics(sentences: list[M2Sentence]) -> dict[str, Any]:
    """Calculate statistics about errors in the corpus."""
    total_sentences = len(sentences)
    total_annotations = sum(len(s.annotations) for s in sentences)
    sentences_with_errors = sum(1 for s in sentences if s.annotations)

    error_type_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}

    for sentence in sentences:
        for ann in sentence.annotations:
            # Count specific error types
            error_type_counts[ann.error_type] = (
                error_type_counts.get(ann.error_type, 0) + 1
            )

            # Count general categories
            category = map_error_to_category(ann.error_type)
            category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "total_sentences": total_sentences,
        "total_annotations": total_annotations,
        "sentences_with_errors": sentences_with_errors,
        "error_rate": sentences_with_errors / total_sentences
        if total_sentences > 0
        else 0,
        "avg_errors_per_sentence": total_annotations / total_sentences
        if total_sentences > 0
        else 0,
        "error_type_distribution": dict(
            sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
        "category_distribution": category_counts,
    }


def main():
    """Test M2 parser on corpus data."""
    import json

    # Parse dev set as example
    m2_path = Path(
        "scripts/training/corpus-raw/user-prompt-final-versions/en-writeandimprove2024-final-versions-dev-sentences.m2"
    )

    if not m2_path.exists():
        print(f"Error: {m2_path} not found")
        return

    print(f"Parsing {m2_path}...")
    sentences = parse_m2_file(m2_path)

    print(f"\nParsed {len(sentences)} sentences")

    # Show example
    print("\n" + "=" * 80)
    print("EXAMPLE SENTENCE WITH ERRORS:")
    print("=" * 80)

    example = next((s for s in sentences if len(s.annotations) > 0), None)
    if example:
        print(f"\nText: {example.text}")
        print(f"Tokens: {example.tokens}")
        print(f"\nAnnotations ({len(example.annotations)}):")
        for ann in example.annotations:
            category = map_error_to_category(ann.error_type)
            print(
                f"  [{ann.start_token}:{ann.end_token}] {ann.error_type} ({category})"
            )
            print(f"    Correction: '{ann.correction}'")

        print(f"\nBIO tags: {create_bio_tags(example)}")

    # Statistics
    print("\n" + "=" * 80)
    print("CORPUS STATISTICS:")
    print("=" * 80)

    stats = extract_error_statistics(sentences)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
