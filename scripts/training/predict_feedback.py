"""
Inference script for AES-FEEDBACK model.

Shows how to use the trained model for predictions with
error detection and attention-based heatmap.
"""

import torch
from typing import Any
from transformers import AutoTokenizer

from feedback_model import FeedbackModel


# CEFR score to level mapping
CEFR_THRESHOLDS = {
    8.5: "C2",
    8.25: "C2",
    7.75: "C1+",
    7.0: "C1",
    6.25: "B2+",
    5.5: "B2",
    4.75: "B1+",
    4.0: "B1",
    3.25: "A2+",
    2.75: "A2",
}


def score_to_cefr(score: float) -> str:
    """Convert numeric score to CEFR level."""
    for threshold, level in CEFR_THRESHOLDS.items():
        if score >= threshold:
            return level
    return "A2"


def predict_with_feedback(
    model: FeedbackModel,
    tokenizer: AutoTokenizer,
    essay_text: str,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """
    Generate comprehensive feedback for an essay.

    Returns:
        {
            "cefr": {"score": float, "level": str},
            "errors": {
                "distribution": {...},
                "problem_spans": [...],
                "heatmap": [...]
            }
        }
    """
    # Tokenize with offset mapping for heatmap
    inputs = tokenizer(
        essay_text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    offset_mapping = inputs["offset_mapping"][0]  # Keep on CPU

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, output_attentions=True)

    # 1. CEFR Score
    cefr_score = outputs["cefr_score"].item()
    cefr_level = score_to_cefr(cefr_score)

    # 2. Error Type Distribution
    error_types = torch.sigmoid(outputs["error_type_logits"]).cpu()
    error_categories = ["grammar", "vocabulary", "mechanics", "fluency", "other"]
    error_distribution = {
        cat: error_types[0, idx].item() for idx, cat in enumerate(error_categories)
    }

    # 3. Error Spans
    span_probs = torch.softmax(outputs["span_logits"], dim=-1).cpu()
    b_error_probs = span_probs[0, :, 0]  # B-ERROR probabilities

    # Find high-confidence error spans
    problem_spans = []
    threshold = 0.7

    for idx, prob in enumerate(b_error_probs):
        if prob > threshold and idx < len(offset_mapping):
            start, end = offset_mapping[idx]
            if start < end:  # Not a special token
                problem_spans.append(
                    {
                        "text": essay_text[start:end],
                        "start_char": int(start),
                        "end_char": int(end),
                        "confidence": float(prob),
                    }
                )

    # 4. Attention Heatmap
    # Average attention across all heads and layers
    attentions = outputs["attentions"]
    last_layer_attention = attentions[-1]  # [batch, num_heads, seq_len, seq_len]
    avg_attention = last_layer_attention.mean(dim=1)[0].mean(dim=0)  # [seq_len]

    # Map tokens to words
    heatmap = []
    for idx, (start, end) in enumerate(offset_mapping):
        if start >= end:  # Special token
            continue

        word = essay_text[start:end]
        attention_score = float(avg_attention[idx])

        heatmap.append(
            {
                "word": word,
                "attention": attention_score,
                "intensity": "high"
                if attention_score > 0.7
                else "medium"
                if attention_score > 0.4
                else "low",
            }
        )

    # Normalize heatmap attention scores
    if heatmap:
        max_attn = max(h["attention"] for h in heatmap)
        for h in heatmap:
            h["attention"] = h["attention"] / max_attn if max_attn > 0 else 0

    return {
        "cefr": {
            "score": cefr_score,
            "level": cefr_level,
        },
        "errors": {
            "distribution": error_distribution,
            "problem_spans": problem_spans,
            "heatmap": heatmap,
        },
        "metadata": {
            "model": "aes-feedback-v1",
            "essay_length": len(essay_text),
            "tokens_used": len(input_ids[0]),
        },
    }


def main():
    """Test inference pipeline."""
    print("=" * 80)
    print("AES-FEEDBACK INFERENCE TEST")
    print("=" * 80)

    # Example essay
    essay = """
    The student have many books in their bag. They goes to library every day
    to study and reading. Its very important for improve english skills.
    """.strip()

    print(f"\nEssay:\n{essay}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Create model (untrained for demo)
    print("Creating model...")
    model = FeedbackModel()
    device = torch.device("cpu")
    model = model.to(device)

    # NOTE: In production, would load trained weights:
    # checkpoint = torch.load("feedback_model_best.pt")
    # model.load_state_dict(checkpoint["model_state_dict"])

    # Predict
    print("\nGenerating feedback...")
    result = predict_with_feedback(model, tokenizer, essay, device)

    # Display results
    print("\n" + "=" * 80)
    print("FEEDBACK RESULTS:")
    print("=" * 80)

    print("\nCEFR Assessment:")
    print(f"  Score: {result['cefr']['score']:.2f}")
    print(f"  Level: {result['cefr']['level']}")

    print("\nError Analysis:")
    print("  Distribution:")
    for cat, value in result["errors"]["distribution"].items():
        print(f"    {cat}: {value:.1%}")

    print(f"\n  Problem Spans: {len(result['errors']['problem_spans'])}")
    for span in result["errors"]["problem_spans"][:3]:
        print(f"    - '{span['text']}' (confidence: {span['confidence']:.2f})")

    print(f"\n  Heatmap Words: {len(result['errors']['heatmap'])}")
    high_attention = [
        h for h in result["errors"]["heatmap"] if h["intensity"] == "high"
    ]
    print(f"    High attention words: {len(high_attention)}")
    for h in high_attention[:5]:
        print(f"      - '{h['word']}' (attention: {h['attention']:.2f})")

    print("\n✅ Inference pipeline validated!")
    print("\nNote: This is using an UNTRAINED model (random weights).")
    print("After training, predictions will be meaningful.")


if __name__ == "__main__":
    main()
