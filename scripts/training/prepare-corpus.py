#!/usr/bin/env python3
"""
Prepare Write and Improve corpus data for training.

Loads corpus data, filters for final versions with human CEFR labels,
converts CEFR to band scores, and saves in format ready for training.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# CEFR to band score mapping (IELTS-aligned)
# Based on CEFR-IELTS correspondence for accurate proficiency prediction
CEFR_TO_SCORE = {
    "A1": 2.0,  # IELTS 0-2.5
    "A1+": 2.5,  # IELTS 2.5
    "A2": 3.0,  # IELTS 3.0-3.5
    "A2+": 3.5,  # IELTS 3.5-4.0
    "B1": 4.5,  # IELTS 4.0-5.0
    "B1+": 5.0,  # IELTS 5.0-5.5
    "B2": 6.0,  # IELTS 5.5-6.5
    "B2+": 6.5,  # IELTS 6.5-7.0
    "C1": 7.5,  # IELTS 7.0-8.0
    "C1+": 8.0,  # IELTS 8.0
    "C2": 8.5,  # IELTS 8.5-9.0
}

# Ordinal class labels (for ordinal regression approach)
CEFR_CLASSES = ["A1", "A1+", "A2", "A2+", "B1", "B1+", "B2", "B2+", "C1", "C1+", "C2"]


def load_corpus_data(corpus_path: str, metadata_path: str) -> pd.DataFrame:
    """Load and merge corpus data."""
    print(f"Loading corpus from {corpus_path}...")
    corpus_df = pd.read_csv(corpus_path, sep="\t", low_memory=False)

    print(f"Loading metadata from {metadata_path}...")
    metadata_df = pd.read_csv(metadata_path, sep="\t", low_memory=False)

    # Merge on essay ID
    # Use split from metadata file (it has train/dev/test), corpus file split may be incomplete
    # Rename split columns to avoid conflict
    metadata_subset = metadata_df[["public_essay_id", "n.edits", "split"]].copy()
    metadata_subset = metadata_subset.rename(columns={"split": "split_meta"})

    merged = corpus_df.merge(
        metadata_subset,
        on="public_essay_id",
        how="left",
    )

    # Use metadata split if available, otherwise fall back to corpus split
    if "split_meta" in merged.columns:
        merged["split"] = merged["split_meta"].fillna(merged.get("split", "unknown"))
        merged = merged.drop(columns=["split_meta"])

    return merged


def filter_essays(df: pd.DataFrame, min_words: int = 50) -> pd.DataFrame:
    """Filter essays for training."""
    print(f"Initial essays: {len(df)}")

    # Filter for final versions with human CEFR labels
    filtered = df[
        (df["is_final_version"] == True)  # noqa: E712
        & (df["humannotator_cefr_level"].notna())
        & (df["humannotator_cefr_level"] != "NA")
        & (df["text"].notna())
        & (df["text"].str.len() > min_words * 5)  # Rough word count estimate
    ].copy()

    print(f"After filtering: {len(filtered)} essays")

    # Show CEFR distribution
    cefr_counts = filtered["humannotator_cefr_level"].value_counts()
    print("\nCEFR distribution:")
    for cefr, count in cefr_counts.items():
        print(f"  {cefr}: {count}")

    return filtered


def convert_cefr_to_score(cefr: str) -> float:
    """Convert CEFR level to band score."""
    return CEFR_TO_SCORE.get(cefr, 4.5)  # Default to B1 if unknown


def convert_cefr_to_ordinal_class(cefr: str) -> int:
    """Convert CEFR level to ordinal class index (0-10)."""
    try:
        return CEFR_CLASSES.index(cefr)
    except ValueError:
        return 4  # Default to B1 (index 4)


def prepare_training_data(
    df: pd.DataFrame, prompts_df: pd.DataFrame | None = None, use_ordinal: bool = False
) -> list[dict]:
    """Prepare training data in format expected by transformers.

    Args:
        df: DataFrame with essay data
        prompts_df: Optional DataFrame with prompt information
        use_ordinal: If True, use ordinal class indices instead of scores
    """
    training_data = []

    for _, row in df.iterrows():
        essay_text = row["text"]
        cefr = row["humannotator_cefr_level"]
        prompt_id = row.get("public_prompt_id", "")

        # Get prompt text if available
        if prompts_df is not None and prompt_id:
            prompt_row = prompts_df[prompts_df["public_prompt_id"] == prompt_id]
            if not prompt_row.empty:
                prompt_text = prompt_row.iloc[0].get(
                    "prompt", f"Essay prompt {prompt_id}"
                )
            else:
                prompt_text = f"Essay prompt {prompt_id}"
        else:
            prompt_text = (
                f"Essay prompt {prompt_id}" if prompt_id else "Write an essay."
            )

        # Format input same as current system
        input_text = f"{prompt_text}\n\n{essay_text}"

        # Convert CEFR to target (either score or ordinal class)
        if use_ordinal:
            target = convert_cefr_to_ordinal_class(cefr)
        else:
            target = convert_cefr_to_score(cefr)

        training_data.append(
            {
                "input": input_text,
                "target": target,  # Either float score or int class
                "cefr": cefr,
                "essay_id": row["public_essay_id"],
                "split": row.get("split", "unknown"),
                "n_edits": row.get("n.edits", 0),
            }
        )

    return training_data


def split_data(data: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split data into train/dev/test based on corpus splits."""
    train_data = [d for d in data if d["split"] == "train"]
    dev_data = [d for d in data if d["split"] == "dev"]
    test_data = [d for d in data if d["split"] == "test"]

    print("\nData splits:")
    print(f"  Train: {len(train_data)}")
    print(f"  Dev: {len(dev_data)}")
    print(f"  Test: {len(test_data)}")

    return train_data, dev_data, test_data


def main():
    parser = argparse.ArgumentParser(description="Prepare corpus data for training")
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=os.path.expanduser(
            "~/Desktop/write-and-improve-corpus-2024-v2/whole-corpus/en-writeandimprove2024-corpus.tsv"
        ),
        help="Path to corpus TSV file",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=os.path.expanduser(
            "~/Desktop/write-and-improve-corpus-2024-v2/user-prompt-final-versions/en-writeandimprove2024-final-versions-m2-essay-info.tsv"
        ),
        help="Path to metadata TSV file",
    )
    parser.add_argument(
        "--prompts-path",
        type=str,
        default=os.path.expanduser(
            "~/Desktop/write-and-improve-corpus-2024-v2/whole-corpus/en-writeandimprove2024-prompts-info.tsv"
        ),
        help="Path to prompts info TSV file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/training/data",
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Minimum word count for essays",
    )

    args = parser.parse_args()

    # Check input files exist
    if not os.path.exists(args.corpus_path):
        print(f"ERROR: Corpus file not found: {args.corpus_path}")
        sys.exit(1)

    if not os.path.exists(args.metadata_path):
        print(f"ERROR: Metadata file not found: {args.metadata_path}")
        sys.exit(1)

    # Load prompts if available
    prompts_df = None
    if os.path.exists(args.prompts_path):
        print(f"Loading prompts from {args.prompts_path}...")
        prompts_df = pd.read_csv(args.prompts_path, sep="\t", low_memory=False)
    else:
        print("WARNING: Prompts file not found, using default prompt text")

    # Load and filter data
    df = load_corpus_data(args.corpus_path, args.metadata_path)
    filtered_df = filter_essays(df, min_words=args.min_words)

    # Prepare training data
    training_data = prepare_training_data(filtered_df, prompts_df)

    # Split data
    train_data, dev_data, test_data = split_data(training_data)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    for split_name, split_data_list in [
        ("train", train_data),
        ("dev", dev_data),
        ("test", test_data),
    ]:
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for item in split_data_list:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {split_name} data to {output_path}")

    # Save metadata
    metadata = {
        "total_essays": len(training_data),
        "train_count": len(train_data),
        "dev_count": len(dev_data),
        "test_count": len(test_data),
        "cefr_distribution": filtered_df["humannotator_cefr_level"]
        .value_counts()
        .to_dict(),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
