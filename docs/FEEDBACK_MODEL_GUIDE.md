# T-AES-FEEDBACK: Technical Guide for Software Engineers

A practical guide to understanding the T-AES-FEEDBACK model - an attention-based essay feedback system that provides actionable error detection beyond CEFR scores.

## Table of Contents

- [Overview](#overview)
- [What Makes It Different](#what-makes-it-different)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Inference & Output](#inference--output)
- [Integration](#integration)
- [Comparison with Other Models](#comparison-with-other-models)

---

## Overview

**T-AES-FEEDBACK** is a multi-task AI model that provides comprehensive essay feedback:

**Outputs:**

1. CEFR proficiency score (A2-C2)
2. **Error span detection** (highlights problematic text)
3. **Error type distribution** (% grammar, vocabulary, etc.)
4. **Attention heatmap** (shows model focus areas)

**Why it exists:** Current models (T-AES-CORPUS, T-AES-ESSAY) only give a score. Learners need to know **where** and **what** to improve.

**Model:** DeBERTa-v3-base fine-tuned with multi-task learning

---

## What Makes It Different

### Current System Limitations

**T-AES-CORPUS** (existing):

```
Input: Essay text
Output: "Your essay is B2 level" (QWK 0.87)
```

**T-AES-ESSAY** (existing):

```
Input: Essay text
Output: Multiple scores (task achievement, coherence, etc.)
```

**Problem:** Neither tells the learner **where errors are** or **what types of errors**.

### T-AES-FEEDBACK Solution

```
Input: Essay text
Output: {
  "cefr_score": 6.2,
  "cefr_level": "B2",
  "error_distribution": {
    "grammar": 0.65,      // 65% of errors are grammar
    "vocabulary": 0.20,   // 20% vocabulary
    "mechanics": 0.10,    // 10% spelling/punctuation
    "fluency": 0.05       // 5% awkward phrasing
  },
  "problem_spans": [
    {
      "text": "have many books",
      "start": 12,
      "end": 27,
      "confidence": 0.87,
      "likely_type": "grammar"  // Suggests SVA issue
    }
  ],
  "attention_heatmap": [
    {"word": "student", "attention": 0.45},
    {"word": "have", "attention": 0.89},  // High = likely error
    ...
  ]
}
```

**Value:** 10x more actionable than just a score!

---

## Data Preparation

### Write & Improve Corpus Annotations

The corpus contains **M2 format** error annotations:

```
S However, in most cases, they depend on the priorities we have.
A 12 12|||M:DET|||the|||REQUIRED|||-NONE-|||0
  ^^    ^^error^^ ^^correction^^
  position  type
```

**What this means:**

- Token 12 has a **missing determiner** (M:DET)
- Should insert "the"
- This is a **required** correction

### Error Categories

We map specific errors to 5 categories:

| M2 Error Type | Category   | Example                     |
| ------------- | ---------- | --------------------------- |
| `R:VERB:SVA`  | grammar    | "The student have" → "has"  |
| `M:DET`       | grammar    | Missing "the"               |
| `R:NOUN:NUM`  | grammar    | "much books" → "many books" |
| `R:PREP`      | grammar    | "depends of" → "depends on" |
| `R:ORTH`      | mechanics  | Spelling errors             |
| `R:PUNCT`     | mechanics  | Punctuation                 |
| `R:WO`        | vocabulary | Word order                  |
| `R:OTHER`     | vocabulary | Word choice                 |
| `U:DET`       | fluency    | Unnecessary word            |

### Data Pipeline

**Step 1: Parse M2 files** ([`parse_m2_annotations.py`](../scripts/training/parse_m2_annotations.py))

```python
# Input: M2 file
# Output: Structured annotations
{
  "text": "The student have many books.",
  "tokens": ["The", "student", "have", "many", "books", "."],
  "annotations": [
    {
      "start_token": 2,
      "end_token": 3,
      "error_type": "R:VERB:SVA",
      "category": "grammar",
      "correction": "has"
    }
  ]
}
```

**Step 2: Create BIO tags** (for token classification)

```python
# BIO = Beginning, Inside, Outside
tokens = ["The", "student", "have", "many", "books", "."]
bio_tags = ["O", "O", "B-ERROR", "O", "O", "O"]
#                      ↑ Error here!
```

**Step 3: Generate enhanced dataset** ([`prepare-enhanced-corpus.py`](../scripts/training/prepare-enhanced-corpus.py))

```python
# Combines CEFR labels + error annotations
{
  "input": "Essay text...",
  "cefr": "B2",
  "target": 6.0,
  "error_count": 11,
  "error_distribution": {
    "grammar": 0.72,
    "vocabulary": 0.09,
    ...
  },
  "annotated_sentences": [...]  // First 5 as training examples
}
```

**Statistics:**

- **Train**: 3,784 essays / 37,486 sentences
- **Dev**: 476 essays / 4,352 sentences
- **Error rate**: 66% of sentences have ≥1 error
- **Category breakdown**: 57% grammar, 17% vocab, 9% mechanics, 9% fluency

---

## Model Architecture

### Multi-Task Learning

**Concept:** Train ONE model to do THREE tasks simultaneously:

```
Essay Text ("The student have many books")
    ↓
┌─────────────────────────────┐
│  DeBERTa-v3 Encoder         │  ← Shared understanding
│  (12 transformer layers)     │
└─────────────────────────────┘
    ↓         ↓           ↓
    │         │           │
Task 1:   Task 2:    Task 3:
CEFR     Error      Error Type
Score    Spans      Distribution
  ↓         ↓           ↓
 6.2    [B,I,O]    [0.65, 0.20, ...]
       for each   grammar, vocab, etc.
        token
```

### Why Multi-Task?

**Benefits:**

1. **Shared features**: Learning to score CEFR helps detect errors (and vice versa)
2. **Regularization**: Prevents overfitting to any single task
3. **Efficiency**: One model, three outputs
4. **Better representations**: Forced to learn richer text understanding

**Trade-off:** May sacrifice 2-3% CEFR accuracy for error detection capability

### Architecture Details

**Base Model:** DeBERTa-v3-base (184M parameters)

- Better than RoBERTa for text understanding
- Disentangled attention mechanism
- 768-dimensional embeddings

**Task Heads:**

```python
class FeedbackModel(nn.Module):
    def __init__(self):
        # Shared encoder
        self.encoder = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")

        # Task 1: CEFR Score (regression)
        self.cefr_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Single score output
        )

        # Task 2: Error Spans (token classification)
        self.span_head = nn.Linear(768, 3)  # B-ERROR, I-ERROR, O

        # Task 3: Error Types (multi-label classification)
        self.error_type_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5)  # 5 error categories
        )
```

**Forward Pass:**

```python
def forward(self, essay_text):
    # 1. Tokenize
    tokens = tokenizer(essay_text, max_length=512)

    # 2. Encode (shared)
    outputs = self.encoder(tokens, output_attentions=True)
    sequence = outputs.last_hidden_state  # [batch, seq_len, 768]
    cls_token = sequence[:, 0]            # [batch, 768]

    # 3. Task outputs
    cefr_score = self.cefr_head(cls_token)           # Use CLS token
    span_logits = self.span_head(sequence)           # All tokens
    error_types = self.error_type_head(cls_token)    # Use CLS token
    attention = outputs.attentions[-1]               # For heatmap

    return {
        "cefr_score": cefr_score,
        "span_logits": span_logits,
        "error_type_logits": error_types,
        "attention_weights": attention
    }
```

### Loss Function

**Multi-task loss** combines three objectives:

```python
total_loss = 1.0 * MSE(predicted_cefr, true_cefr) +        # Primary task
             0.5 * CrossEntropy(span_logits, bio_tags) +   # Error detection
             0.3 * BCELoss(error_types, true_types)        # Error classification
```

**Weights:**

- **CEFR = 1.0**: Primary task (maintain accuracy)
- **Spans = 0.5**: Important but secondary
- **Types = 0.3**: Nice to have

**Why weighted?** CEFR accuracy is critical; error detection is enhancement.

---

## Training Process

### Hardware: Modal GPU

```python
@modal.app.function(
    gpu="A10G",              # NVIDIA A10G (24GB VRAM)
    timeout=7200,            # 2 hours
    secrets=[hf_token],
)
def train_feedback_model():
    # Training code here
```

**Why Modal?**

- No local GPU needed
- Pay-per-use (~$1-2/hour)
- Easy deployment after training

### Training Configuration

```yaml
model: microsoft/deberta-v3-base
max_length: 512
batch_size: 16
learning_rate: 2e-5
epochs: 10-15
warmup_steps: 500

# Multi-task loss weights
cefr_weight: 1.0
span_weight: 0.5
error_type_weight: 0.3

# Early stopping
patience: 3 # Stop if no improvement for 3 epochs
metric: cefr_qwk # Primary metric to monitor
```

### Training Loop

```python
for epoch in range(15):
    for batch in train_loader:
        # Forward pass
        outputs = model(batch["input_ids"], batch["attention_mask"])

        # Compute multi-task loss
        loss, metrics = loss_fn(outputs, batch["targets"])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log metrics
        print(f"CEFR loss: {metrics['cefr']:.4f}")
        print(f"Span loss: {metrics['span']:.4f}")
        print(f"Type loss: {metrics['error_type']:.4f}")

    # Validation
    dev_metrics = evaluate(dev_loader)

    # Early stopping
    if dev_metrics["cefr_qwk"] > best_qwk:
        save_checkpoint()
        best_qwk = dev_metrics["cefr_qwk"]
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 3:
            print("Early stopping!")
            break
```

### Expected Performance

**Targets:**

| Metric               | Target | Current Baseline    |
| -------------------- | ------ | ------------------- |
| CEFR QWK             | ≥0.82  | 0.87 (T-AES-CORPUS) |
| CEFR MAE             | ≤0.40  | 0.32 (T-AES-CORPUS) |
| Span Detection F1    | ≥0.70  | N/A (new)           |
| Error Type Precision | ≥0.60  | N/A (new)           |

**Acceptable trade-off:** 2-5% CEFR accuracy drop for gaining error detection.

---

## Inference & Output

### Prediction Pipeline

```python
def predict_with_feedback(essay_text: str):
    # 1. Tokenize
    inputs = tokenizer(
        essay_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        return_offsets_mapping=True
    )

    # 2. Model forward
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 3. Post-process outputs
    cefr_score = outputs["cefr_score"].item()

    # Spans: Get tokens with high error probability
    span_probs = torch.softmax(outputs["span_logits"], dim=-1)
    error_mask = span_probs[:, :, 0] > 0.7  # B-ERROR > 70%

    # Types: Get error distribution
    error_types = torch.sigmoid(outputs["error_type_logits"])

    # Attention: Average across heads for heatmap
    attention = outputs["attention_weights"][-1].mean(dim=1)

    return {
        "cefr": {
            "score": cefr_score,
            "level": score_to_cefr(cefr_score)
        },
        "errors": {
            "distribution": {
                "grammar": error_types[0, 0].item(),
                "vocabulary": error_types[0, 1].item(),
                ...
            },
            "spans": extract_spans(error_mask, inputs),
            "heatmap": create_heatmap(attention, inputs)
        }
    }
```

### Heatmap Generation

**Concept:** Show which words the model paid attention to (likely errors)

```python
def create_heatmap(attention, inputs):
    # Average attention across all tokens
    avg_attention = attention.mean(dim=1)[0]  # [seq_len]

    # Map tokens back to words
    words = []
    word_attentions = []

    for idx, (start, end) in enumerate(inputs["offset_mapping"][0]):
        if start == end:  # Special token
            continue

        word = essay_text[start:end]
        attn = avg_attention[idx].item()

        words.append(word)
        word_attentions.append(attn)

    # Normalize 0-1
    max_attn = max(word_attentions)
    normalized = [a / max_attn for a in word_attentions]

    return [
        {"word": w, "attention": a}
        for w, a in zip(words, normalized)
    ]
```

### API Response Format

```json
{
  "cefr": {
    "score": 6.24,
    "level": "B2"
  },
  "errors": {
    "total_count": 8,
    "distribution": {
      "grammar": 0.625,
      "vocabulary": 0.25,
      "mechanics": 0.125,
      "fluency": 0.0,
      "other": 0.0
    },
    "problem_spans": [
      {
        "text": "have many books",
        "start_char": 12,
        "end_char": 27,
        "confidence": 0.87,
        "likely_type": "grammar"
      }
    ],
    "heatmap": [
      { "word": "The", "attention": 0.12, "intensity": "low" },
      { "word": "student", "attention": 0.45, "intensity": "medium" },
      { "word": "have", "attention": 0.89, "intensity": "high" },
      { "word": "many", "attention": 0.34, "intensity": "medium" },
      { "word": "books", "attention": 0.28, "intensity": "medium" }
    ]
  },
  "metadata": {
    "model": "t-aes-feedback-v1",
    "inference_time_ms": 245
  }
}
```

---

## Integration

### Frontend Changes

**Heatmap Visualization:**

```typescript
// Color words by attention intensity
const getColor = (attention: number) => {
  if (attention > 0.7) return 'bg-red-200';      // High attention
  if (attention > 0.4) return 'bg-yellow-100';   // Medium
  return 'bg-green-50';                          // Low (likely correct)
};

<div className="essay-heatmap">
  {heatmap.map(({word, attention}) => (
    <span className={getColor(attention)}>
      {word}
    </span>
  ))}
</div>
```

**Error Distribution Chart:**

```typescript
<ErrorBreakdownChart
  grammar={0.625}
  vocabulary={0.250}
  mechanics={0.125}
  fluency={0.000}
/>
```

### Modal Deployment

```python
# Deploy as separate service
@modal.app.function(
    image=modal.Image.debian_slim()
        .pip_install("transformers", "torch"),
    gpu="T4",  # Smaller GPU for inference
    keep_warm=1,  # Keep one instance running
)
@modal.web_endpoint(method="POST")
def score_with_feedback(request):
    essay = request.json()["text"]
    result = predict_with_feedback(essay)
    return result
```

**URL**: `https://rob-gilks--writeo-feedback-fastapi-app.modal.run`

---

## Comparison with Other Models

### Feature Matrix

| Feature                      | T-AES-CORPUS  | T-AES-ESSAY     | T-AES-FEEDBACK    |
| ---------------------------- | ------------- | --------------- | ----------------- |
| **CEFR Score**               | ✅ (QWK 0.87) | ✅              | ✅ (target 0.82+) |
| **Multi-dimensional scores** | ❌            | ✅              | ✅                |
| **Error detection**          | ❌            | ❌              | ✅                |
| **Error types**              | ❌            | ❌              | ✅                |
| **Attention heatmap**        | ❌            | ❌              | ✅                |
| **Speed**                    | Fast (~200ms) | Slow (~2s)      | Medium (~300ms)   |
| **Model size**               | 125M params   | Multiple models | 184M params       |

### Use Cases

**T-AES-CORPUS**: Quick overall assessment

- ✅ Fast, accurate CEFR scoring
- ❌ No actionable feedback

**T-AES-ESSAY**: Detailed rubric scoring

- ✅ Multiple dimensions (coherence, task achievement, etc.)
- ❌ No error-level feedback

**T-AES-FEEDBACK**: Comprehensive feedback

- ✅ CEFR score + error detection + heatmap
- ✅ Actionable insights for learners
- ⚠️ Slightly slower, may have 2-5% lower CEFR accuracy

### Ensemble Strategy

**Best of all worlds:**

```typescript
async function getCompleteFeedback(essay: string) {
  // Run all three in parallel
  const [corpus, essay, feedback] = await Promise.all([
    corpusService.score(essay), // Fast, accurate CEFR
    essayService.grade(essay), // Detailed rubric scores
    feedbackService.analyze(essay), // Error detection + heatmap
  ]);

  return {
    cefr_score: corpus.score, // Most accurate
    dimensions: essay.dimensions, // Rubric feedback
    errors: feedback.errors, // Error insights
    heatmap: feedback.heatmap, // Visual feedback
  };
}
```

---

## Key Takeaways

1. **Purpose**: Provide **actionable error feedback** beyond just scores
2. **Data**: Uses Write & Improve M2 annotations (66% sentences have errors)
3. **Architecture**: Multi-task DeBERTa-v3 (CEFR + spans + types)
4. **Trade-off**: May sacrifice 2-5% CEFR accuracy for error detection
5. **Output**: Score + error distribution + heatmap + problem spans
6. **Deployment**: Modal GPU for training, separate inference service
7. **Value**: 10x more useful feedback for learners

**Status:** Phase 1 & 2 complete (data + architecture). Phase 3 (training) next.

---

## Further Reading

- **Implementation code**: [`feedback_model.py`](../scripts/training/feedback_model.py)
- **Data pipeline**: [`parse_m2_annotations.py`](../scripts/training/parse_m2_annotations.py)
- **Original corpus guide**: [CORPUS_MODEL_GUIDE.md](CORPUS_MODEL_GUIDE.md)
- **DeBERTa paper**: [DeBERTaV3](https://arxiv.org/abs/2111.09543)
- **Multi-task learning**: [An Overview](https://ruder.io/multi-task/)
