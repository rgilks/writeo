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

        # Error span labels (Task 2) - Use REAL M2 annotations!
        span_labels = self._create_span_labels(example)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "cefr_score": cefr_score,
            "span_labels": span_labels,
            "error_type_labels": error_types,
        }

    def _create_span_labels(self, example: dict[str, Any]) -> torch.Tensor:
        """
        Create BIO labels from M2 annotated sentences.

        Uses actual error positions from corpus annotations.
        """
        from parse_m2_annotations import M2Annotation, align_m2_to_subword_tokens

        # Get annotated sentences (first 5 per essay)
        annotated_sents = example.get("annotated_sentences", [])

        if not annotated_sents:
            # No annotations - return all "O" (no errors)
            return torch.full((self.max_length,), -100, dtype=torch.long)

        # Combine first 5 sentences with their annotations
        combined_text_parts = []
        all_annotations = []

        for sent_data in annotated_sents[:5]:  # Limit to 5 sentences
            sent_text = sent_data.get("text", "")
            combined_text_parts.append(sent_text)

            # Convert annotations with adjusted offsets
            for ann_data in sent_data.get("annotations", []):
                # Create M2Annotation from dict
                m2_ann = M2Annotation(
                    start_token=ann_data["start_token"],
                    end_token=ann_data["end_token"],
                    error_type=ann_data.get("error_type", ""),
                    correction=ann_data.get("correction", ""),
                    required=ann_data.get("required", True),
                )
                all_annotations.append(m2_ann)

        if not combined_text_parts:
            return torch.full((self.max_length,), -100, dtype=torch.long)

        # Combine sentences with spaces
        combined_text = " ".join(combined_text_parts)

        # Get BIO tags using alignment function
        bio_tags_str = align_m2_to_subword_tokens(
            combined_text, all_annotations, self.tokenizer
        )

        # Convert to numeric labels
        label_map = {"O": 0, "B-ERROR": 1, "I-ERROR": 2}
        bio_labels = [label_map.get(tag, 0) for tag in bio_tags_str]

        # Truncate or pad to max_length
        if len(bio_labels) > self.max_length:
            bio_labels = bio_labels[: self.max_length]
        else:
            # Pad with -100 (ignored in loss)
            bio_labels += [-100] * (self.max_length - len(bio_labels))

        return torch.tensor(bio_labels, dtype=torch.long)


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
        Path("data-enhanced/dev-enhanced.jsonl"),
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
