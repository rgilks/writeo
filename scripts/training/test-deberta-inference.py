"""
Test inference on trained AES-DEBERTA model.
"""

import modal

# Volume with trained model
model_volume = modal.Volume.from_name("writeo-deberta-models")

app = modal.App("writeo-test-deberta")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.6.0",
    "transformers>=4.40.0",
    "sentencepiece>=0.1.99",
    "safetensors>=0.4.2",
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": model_volume},
    timeout=300,
)
def test_inference():
    """Test model inference with sample essays."""
    import os
    import torch
    from transformers import AutoTokenizer

    MODEL_PATH = "/vol/models/deberta-v3-aes"
    MODEL_NAME = "microsoft/deberta-v3-large"

    print("=" * 60)
    print("🧪 Testing AES-DEBERTA Inference")
    print("=" * 60)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        return {"error": f"Model not found at {MODEL_PATH}"}

    print(f"\n✅ Model found at {MODEL_PATH}")
    print(f"   Files: {os.listdir(MODEL_PATH)}")

    # Load tokenizer
    print(f"\n📦 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Define model classes inline to avoid import issues
    import torch.nn as nn
    from transformers import DebertaV2Model

    class AttentionPooling(nn.Module):
        def __init__(self, hidden_size, dropout=0.1):
            super().__init__()
            self.query = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, hidden_states, attention_mask):
            import torch.nn.functional as F

            scores = self.query(hidden_states).squeeze(-1)
            scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
            weights = F.softmax(scores, dim=1)
            weights = self.dropout(weights)
            return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)

    class CORNHead(nn.Module):
        def __init__(self, hidden_size, num_thresholds=7):
            super().__init__()
            self.num_thresholds = num_thresholds
            self.shared = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.threshold_classifiers = nn.Linear(256, num_thresholds)

        def forward(self, pooled_output):
            shared = self.shared(pooled_output)
            return self.threshold_classifiers(shared)

        def logits_to_score(self, logits, min_score=2.0):
            probs = torch.sigmoid(logits)
            return min_score + probs.sum(dim=1)

    class DimensionHead(nn.Module):
        def __init__(self, hidden_size, dropout=0.1):
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        def forward(self, pooled_output):
            return self.head(pooled_output)

    class DeBERTaAESModel(nn.Module):
        def __init__(
            self,
            model_name="microsoft/deberta-v3-large",
            num_cefr_thresholds=7,
            dropout=0.1,
        ):
            super().__init__()
            self.encoder = DebertaV2Model.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.pooling = AttentionPooling(hidden_size, dropout=dropout)
            self.ta_head = DimensionHead(hidden_size, dropout)
            self.cc_head = DimensionHead(hidden_size, dropout)
            self.vocab_head = DimensionHead(hidden_size, dropout)
            self.grammar_head = DimensionHead(hidden_size, dropout)
            self.cefr_head = CORNHead(hidden_size, num_cefr_thresholds)
            self.hidden_size = hidden_size

        def forward(self, input_ids, attention_mask, output_attentions=False):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = outputs.last_hidden_state
            pooled = self.pooling(hidden_states, attention_mask)
            ta_score = self.ta_head(pooled).squeeze(-1)
            cc_score = self.cc_head(pooled).squeeze(-1)
            vocab_score = self.vocab_head(pooled).squeeze(-1)
            grammar_score = self.grammar_head(pooled).squeeze(-1)
            cefr_logits = self.cefr_head(pooled)
            cefr_score = self.cefr_head.logits_to_score(cefr_logits)
            overall = (ta_score + cc_score + vocab_score + grammar_score) / 4
            return {
                "ta": ta_score,
                "cc": cc_score,
                "vocab": vocab_score,
                "grammar": grammar_score,
                "cefr_logits": cefr_logits,
                "cefr_score": cefr_score,
                "overall": overall,
            }

    # Load model
    print(f"\n🧠 Loading model from {MODEL_PATH}")
    model = DeBERTaAESModel(model_name=MODEL_NAME)
    model_state = torch.load(
        os.path.join(MODEL_PATH, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(model_state)
    model.eval()
    model = model.cuda()
    print("✅ Model loaded successfully")

    # Test essays
    test_essays = [
        {
            "level": "A2 (beginner)",
            "text": "I like dog. Dog is my friend. I have one dog. His name is Max. Max is brown dog. I love Max very much.",
        },
        {
            "level": "B2 (intermediate)",
            "text": "In my opinion, technology has significantly improved our lives in many ways. People can now communicate with others around the world instantly. However, there are some disadvantages such as privacy concerns and the impact on face-to-face interactions.",
        },
        {
            "level": "C1 (advanced)",
            "text": "The proliferation of artificial intelligence in contemporary society raises profound ethical questions regarding autonomy, employment, and the very nature of human cognition. While proponents argue that AI offers unprecedented opportunities for solving complex global challenges, critics contend that the technology's rapid advancement outpaces our ability to develop appropriate regulatory frameworks.",
        },
    ]

    print("\n" + "=" * 60)
    print("📝 Testing with sample essays:")
    print("=" * 60)

    results = []
    for essay in test_essays:
        # Tokenize
        encoded = tokenizer(
            essay["text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()

        # Inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        # Extract scores
        result = {
            "expected_level": essay["level"],
            "TA": round(outputs["ta"].item(), 1),
            "CC": round(outputs["cc"].item(), 1),
            "Vocab": round(outputs["vocab"].item(), 1),
            "Grammar": round(outputs["grammar"].item(), 1),
            "Overall": round(outputs["overall"].item(), 1),
            "CEFR_score": round(outputs["cefr_score"].item(), 1),
        }
        results.append(result)

        print(f"\n📄 Essay ({essay['level']}):")
        print(f'   "{essay["text"][:80]}..."')
        print(
            f"   → TA={result['TA']}, CC={result['CC']}, Vocab={result['Vocab']}, Grammar={result['Grammar']}"
        )
        print(f"   → Overall={result['Overall']}, CEFR={result['CEFR_score']}")

    print("\n" + "=" * 60)
    print("✅ Inference test complete!")
    print("=" * 60)

    return {"status": "success", "results": results}


@app.local_entrypoint()
def main():
    """Run inference test."""
    result = test_inference.remote()
    print(f"\nTest result: {result}")


if __name__ == "__main__":
    main()
