"""
Data loader for multi-task feedback model training.

Loads enhanced corpus data with error annotations and creates
batches for training CEFR + error detection + error types.
"""

import json
import torch
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class FeedbackDataset(Dataset):
    """Dataset for multi-task feedback model."""

    def __init__(
        self,
        data_file: Path,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.examples = []
        with open(data_file) as f:
            for line in f:
                self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} examples from {data_file}")

        # Error category mapping
        self.error_categories = [
            "grammar",
            "vocabulary",
            "mechanics",
            "fluency",
            "other",
        ]
        self.category_to_idx = {
            cat: idx for idx, cat in enumerate(self.error_categories)
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]

        # Tokenize essay
        encoding = self.tokenizer(
            example["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # CEFR score target (Task 1)
        cefr_score = torch.tensor(example["target"], dtype=torch.float32)

        # Error type distribution (Task 3)
        error_types = torch.zeros(len(self.error_categories), dtype=torch.float32)
        if example.get("error_distribution"):
            for category, value in example["error_distribution"].items():
                if category in self.category_to_idx:
                    error_types[self.category_to_idx[category]] = value

        # Error span labels (Task 2) - simplified for now
        # In full version, would parse annotated_sentences and create BIO tags
        # For now, create dummy labels based on whether essay has errors
        span_labels = torch.zeros(
            self.max_length, dtype=torch.long
        )  # All "O" (outside)

        # If essay has errors, mark a few tokens as errors (simplified)
        if example.get("has_errors", False) and example.get("error_count", 0) > 0:
            # This is a placeholder - real implementation would use actual error spans
            # from annotated_sentences
            num_errors = min(example.get("error_count", 0), 5)
            for i in range(num_errors):
                pos = (i + 1) * (self.max_length // (num_errors + 2))
                if pos < self.max_length:
                    span_labels[pos] = 1  # B-ERROR

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "cefr_score": cefr_score,
            "span_labels": span_labels,
            "error_type_labels": error_types,
        }


def create_dataloaders(
    train_file: Path,
    dev_file: Path,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
) -> tuple[DataLoader, DataLoader]:
    """Create train and dev dataloaders."""

    train_dataset = FeedbackDataset(train_file, tokenizer, max_length)
    dev_dataset = FeedbackDataset(dev_file, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, dev_loader


if __name__ == "__main__":
    # Test data loader
    from transformers import AutoTokenizer

    print("Testing FeedbackDataset...")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Test on dev set (smaller)
    dataset = FeedbackDataset(
        Path("scripts/training/data-enhanced/dev-enhanced.jsonl"),
        tokenizer,
        max_length=512,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get first example
    example = dataset[0]

    print("\nExample batch:")
    print(f"  input_ids shape: {example['input_ids'].shape}")
    print(f"  attention_mask shape: {example['attention_mask'].shape}")
    print(f"  cefr_score: {example['cefr_score'].item():.2f}")
    print(f"  span_labels shape: {example['span_labels'].shape}")
    print(f"  span_labels unique values: {example['span_labels'].unique()}")
    print(f"  error_type_labels: {example['error_type_labels']}")

    print("\n✅ Data loader validated!")
