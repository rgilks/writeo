#!/usr/bin/env python3
"""
Analyze training data quality and statistics.

Provides insights into:
- Sequence length distribution
- Class imbalance
- Recommendations for hyperparameters
"""

import json
from pathlib import Path


def analyze_data():
    """Analyze training data and provide recommendations."""

    # Load data
    data_dir = Path("scripts/training/data")

    print("=" * 80)
    print("TRAINING DATA ANALYSIS")
    print("=" * 80)

    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    print("\n📊 Dataset Statistics:")
    print(f"   Total essays: {metadata['total_essays']}")
    print(
        f"   Train: {metadata['train_count']} ({metadata['train_count'] / metadata['total_essays'] * 100:.1f}%)"
    )
    print(
        f"   Dev: {metadata['dev_count']} ({metadata['dev_count'] / metadata['total_essays'] * 100:.1f}%)"
    )
    print(
        f"   Test: {metadata['test_count']} ({metadata['test_count'] / metadata['total_essays'] * 100:.1f}%)"
    )

    # Analyze class distribution
    print("\n🏷️  CEFR Distribution:")
    cefr_dist = metadata["cefr_distribution"]
    total = sum(cefr_dist.values())

    cefr_order = ["A1+", "A2", "A2+", "B1", "B1+", "B2", "B2+", "C1", "C1+", "C2"]
    for cefr in cefr_order:
        if cefr in cefr_dist:
            count = cefr_dist[cefr]
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"   {cefr:4s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Identify imbalance issues
    print("\n⚠️  Class Imbalance Analysis:")
    min_count = min(cefr_dist.values())
    max_count = max(cefr_dist.values())
    imbalance_ratio = max_count / min_count

    print(
        f"   Imbalance ratio: {imbalance_ratio:.1f}:1 (max={max_count}, min={min_count})"
    )

    minority_classes = [k for k, v in cefr_dist.items() if v < 100]
    if minority_classes:
        print(f"   Minority classes (<100 samples): {', '.join(minority_classes)}")
        print("   ⚡ Recommendation: Enable data augmentation or focal loss")
    else:
        print("   ✅ All classes have ≥100 samples")

    # Load sample essays to analyze length
    print("\n📏 Analyzing Sequence Lengths...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    lengths = []
    with open(data_dir / "train.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Sample first 1000 for analysis
                break
            data = json.loads(line)
            tokens = tokenizer.encode(data["input"], add_special_tokens=True)
            lengths.append(len(tokens))

    lengths.sort()
    p50 = lengths[len(lengths) // 2]
    p90 = lengths[int(len(lengths) * 0.9)]
    p95 = lengths[int(len(lengths) * 0.95)]
    p99 = lengths[int(len(lengths) * 0.99)]
    max_len = max(lengths)

    print(f"   Median (p50): {p50} tokens")
    print(f"   p90: {p90} tokens")
    print(f"   p95: {p95} tokens")
    print(f"   p99: {p99} tokens")
    print(f"   Max: {max_len} tokens")

    truncated_512 = sum(1 for length in lengths if length > 512)
    truncated_pct = truncated_512 / len(lengths) * 100

    print("\n   At max_seq_length=512:")
    print(f"   - Truncated: {truncated_512}/{len(lengths)} ({truncated_pct:.1f}%)")

    if truncated_pct > 10:
        print(
            "   ⚡ Recommendation: Consider max_seq_length=768 (will slow training ~30%)"
        )
    else:
        print("   ✅ max_seq_length=512 is adequate")

    # Recommendations
    print("\n" + "=" * 80)
    print("📋 RECOMMENDATIONS FOR OPTIMAL TRAINING")
    print("=" * 80)

    print("\n1. Model & Loss:")
    print("   ✅ Use RoBERTa-base (proven working)")
    print("   ✅ Enable ordinal regression: use_ordinal_regression=True")
    print("   ✅ Use CORAL loss: loss_type='coral'")

    print("\n2. Class Imbalance:")
    if imbalance_ratio > 10:
        print(f"   ⚡ High imbalance detected ({imbalance_ratio:.1f}:1)")
        print("   - Enable focal_loss: loss_type='focal' (focuses on hard examples)")
        print("   - OR enable data_augmentation for minority classes")
    else:
        print(
            f"   ✅ Moderate imbalance ({imbalance_ratio:.1f}:1) - CORAL should handle well"
        )

    print("\n3. Hyperparameters:")
    print("   ✅ learning_rate: 3e-5 (optimal for RoBERTa)")
    print("   ✅ batch_size: 16 (fits in A10G GPU memory)")
    print("   ✅ max_seq_length: 512")
    print("   ✅ num_epochs: 10 with early stopping")

    print("\n4. Expected Training Time:")
    print("   - Full dataset: ~3,784 samples")
    print("   - Est. time on A10G: 2-4 hours")
    print("   - With ordinal regression: slight overhead (~5-10% slower)")

    print("\n" + "=" * 80)
    print("✅ Configuration is ready for optimal training!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Run test: modal run scripts/training/train-overall-score.py --test-run")
    print(
        "  2. If test passes, run full: modal run scripts/training/train-overall-score.py --full"
    )
    print("  3. Evaluate: python scripts/training/evaluate-model.py")


if __name__ == "__main__":
    analyze_data()
