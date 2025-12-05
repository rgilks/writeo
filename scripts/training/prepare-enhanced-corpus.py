#!/usr/bin/env python3
"""
Prepare enhanced corpus data with error annotations for T-AES-FEEDBACK training.

Combines:
1. Original corpus (CEFR labels)
2. M2 error annotations (error types, spans)
3. BIO tags for token-level error detection
"""

import json
from pathlib import Path
from typing import Any

from parse_m2_annotations import (
    parse_m2_file,
    map_error_to_category,
    create_bio_tags,
    M2Sentence,
)


def load_original_corpus_data(split: str) -> dict[str, dict]:
    """Load original corpus data by essay ID."""
    data_file = Path(f"scripts/training/data/{split}.jsonl")

    essays_by_id = {}
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            essays_by_id[item["essay_id"]] = item

    return essays_by_id


def align_m2_with_corpus(
    m2_sentences: list[M2Sentence], corpus_data: dict[str, dict], ids_file: Path
) -> list[dict[str, Any]]:
    """
    Align M2 sentence-level annotations with essay-level corpus data.

    The .ids file maps sentence indices to essay IDs.
    """
    # Load sentence-to-essay mapping
    essay_ids = []
    with open(ids_file) as f:
        for line in f:
            essay_ids.append(line.strip())

    if len(essay_ids) != len(m2_sentences):
        print(f"Warning: {len(essay_ids)} IDs != {len(m2_sentences)} sentences")

    # Group sentences by essay
    essays_with_annotations: dict[str, list[tuple[int, M2Sentence]]] = {}
    for idx, (essay_id, sentence) in enumerate(zip(essay_ids, m2_sentences)):
        if essay_id not in essays_with_annotations:
            essays_with_annotations[essay_id] = []
        essays_with_annotations[essay_id].append((idx, sentence))

    # Create enhanced dataset
    enhanced_data = []
    for essay_id, sentences in essays_with_annotations.items():
        if essay_id not in corpus_data:
            continue  # Skip if no CEFR label

        essay = corpus_data[essay_id]

        # Aggregate error statistics for the essay
        total_errors = sum(len(s.annotations) for _, s in sentences)
        error_categories = {
            "grammar": 0,
            "vocabulary": 0,
            "mechanics": 0,
            "fluency": 0,
            "other": 0,
        }

        for _, sentence in sentences:
            for ann in sentence.annotations:
                category = map_error_to_category(ann.error_type)
                error_categories[category] += 1

        # Normalize to percentages
        error_percentages = {
            cat: count / total_errors if total_errors > 0 else 0.0
            for cat, count in error_categories.items()
        }

        # Create BIO tags for all sentences
        sentence_annotations = []
        for idx, sentence in sentences:
            if sentence.annotations:
                sentence_annotations.append(
                    {
                        "text": sentence.text,
                        "tokens": sentence.tokens,
                        "bio_tags": create_bio_tags(
                            sentence
                        ),  # Space-delimited BIO tags
                        "annotations": [  # Raw M2 annotations for subword alignment
                            {
                                "start_token": ann.start_token,
                                "end_token": ann.end_token,
                                "error_type": ann.error_type,
                                "correction": ann.correction,
                                "required": ann.required,
                            }
                            for ann in sentence.annotations
                        ],
                        "errors": [
                            {
                                "start": ann.start_token,
                                "end": ann.end_token,
                                "type": ann.error_type,
                                "category": map_error_to_category(ann.error_type),
                                "correction": ann.correction,
                            }
                            for ann in sentence.annotations
                        ],
                    }
                )

        enhanced_data.append(
            {
                **essay,  # Include original fields (input, target, cefr, etc.)
                "error_count": total_errors,
                "error_distribution": error_percentages,
                "has_errors": total_errors > 0,
                "annotated_sentences": sentence_annotations[
                    :5
                ],  # Store first 5 as examples
            }
        )

    return enhanced_data


def main():
    """Create enhanced dataset with error annotations."""
    print("=" * 80)
    print("CREATING ENHANCED DATASET WITH ERROR ANNOTATIONS")
    print("=" * 80)

    corpus_raw = Path("scripts/training/corpus-raw/user-prompt-final-versions")
    output_dir = Path("scripts/training/data-enhanced")
    output_dir.mkdir(exist_ok=True)

    for split in ["train", "dev"]:
        print(f"\n{'=' * 80}")
        print(f"Processing {split} set...")
        print(f"{'=' * 80}")

        # Load M2 annotations
        m2_file = (
            corpus_raw / f"en-writeandimprove2024-final-versions-{split}-sentences.m2"
        )
        ids_file = (
            corpus_raw / f"en-writeandimprove2024-final-versions-{split}-sentences.ids"
        )

        if not m2_file.exists():
            print(f"Skipping {split}: {m2_file} not found")
            continue

        print(f"Parsing {m2_file.name}...")
        m2_sentences = parse_m2_file(m2_file)
        print(f"  Loaded {len(m2_sentences)} sentences")

        # Load original corpus data
        print("Loading original corpus data...")
        corpus_data = load_original_corpus_data(split)
        print(f"  Loaded {len(corpus_data)} essays")

        # Align and merge
        print("Aligning M2 annotations with corpus...")
        enhanced_data = align_m2_with_corpus(m2_sentences, corpus_data, ids_file)
        print(f"  Created {len(enhanced_data)} enhanced essays")

        # Save
        output_file = output_dir / f"{split}-enhanced.jsonl"
        with open(output_file, "w") as f:
            for item in enhanced_data:
                f.write(json.dumps(item) + "\n")

        print(f"✅ Saved to {output_file}")

        # Show example
        if enhanced_data:
            example = next(
                (e for e in enhanced_data if e["has_errors"]), enhanced_data[0]
            )
            print("\nExample essay:")
            print(f"  CEFR: {example['cefr']}")
            print(f"  Errors: {example['error_count']}")
            print("  Error distribution:")
            for cat, pct in example["error_distribution"].items():
                if pct > 0:
                    print(f"    {cat}: {pct:.1%}")

    print(f"\n{'=' * 80}")
    print("✅ Enhanced dataset created successfully!")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {output_dir}")
    print("Next step: Train multi-task model with error detection")


if __name__ == "__main__":
    main()
