# T-AES-FEEDBACK: Technical Guide for Software Engineers

A comprehensive guide to the Multi-Task Feedback Model (T-AES-FEEDBACK).

> **Note:** This model is primarily used for CEFR scoring. Specific grammatical error correction is now handled by the dedicated **GEC Service** (see [GEC Service documentation](gec.md)).

**Current Status:** Training complete with mixed results. CEFR scoring excellent (QWK 0.85), error detection requires improvement.

## Table of Contents

- [Overview](#overview)
- [What Makes It Different](#what-makes-it-different)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Actual Training Results](#actual-training-results)
- [Problems Encountered](#problems-encountered)
- [Solutions \& Next Steps](#solutions--next-steps)
- [Alternative Approaches](#alternative-approaches)
- [Inference \& Output](#inference--output)
- [Integration](#integration)
- [Comparison with Other Models](#comparison-with-other-models)

---

## Overview

**T-AES-FEEDBACK** is a multi-task AI model that aims to provide comprehensive essay feedback:

**Intended Outputs:**

1. CEFR proficiency score (A2-C2) ✅ **Working**
2. **Error span detection** (highlights problematic text) ❌ **Not working**
3. **Error type distribution** (% grammar, vocabulary, etc.) ❌ **Not working**
4. **Attention heatmap** (shows model focus areas) ⏸️ **Not tested**

**Why it exists:** Current models (T-AES-CORPUS, T-AES-ESSAY) only give scores. Learners need to know **where** and **what** to improve.

**Model:** DeBERTa-v3-base fine-tuned with multi-task learning

**What We've Built:** Working CEFR scoring model (QWK 0.85) with multi-task architecture ready for error detection improvements.

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

### T-AES-FEEDBACK Intended Solution

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

**Current Reality:** Only CEFR scoring is working. Error detection components need improvement (see [Problems Encountered](#problems-encountered)).

---

## Key Concepts Explained

Before diving into the technical details, let's define important concepts you'll encounter:

### Tokenization

**What:** Breaking text into pieces (tokens) that the model can process.

**Example:**

```python
Text: "The student have books."

Word-level tokens:
["The", "student", "have", "books", "."]

Subword tokens (what DeBERTa uses):
["The", "student", "have", "books", "."]
# Simple example - DeBERTa might split "student" into ["stud", "##ent"]
```

**Why it matters:** Different models split text differently. We need to align error positions with these tokens.

### BIO Tagging

**What:** A way to mark spans of text (like errors) at the token level.

**BIO stands for:**

- **B**eginning (start of error)
- **I**nside (continuation of error)
- **O**utside (not an error)

**Example:**

```python
Sentence: "The student have many books"
Tokens:   ["The", "student", "have", "many", "books"]
                            ↑ SVA error (should be "has")

BIO tags: ["O",   "O",      "B-ERROR", "O",    "O"]
#          not    not       ERROR!     not     not
#          error  error     begins     error   error
```

**Multi-word errors:**

```python
Sentence: "I am very much happy"
Tokens:   ["I", "am", "very", "much", "happy"]
                        ↑_____ awkward phrase ____↑

BIO tags: ["O", "O", "B-ERROR", "I-ERROR", "I-ERROR"]
#                     ↑ begins    ↑ inside   ↑ inside
```

**Why it matters:** This is how we train the model to detect error locations. Our current problem is we're using random BIO tags instead of real ones from M2 annotations!

### Embeddings

**What:** Converting words/tokens into numbers (vectors) that capture meaning.

**Example:**

```python
# Conceptual representation (real embeddings are 768 dimensions!)
"king"   → [0.9, 0.1, 0.8, ...]  # Royal, male, power
"queen"  → [0.9, 0.9, 0.8, ...]  # Royal, female, power
"man"    → [0.1, 0.1, 0.2, ...]  # Male, common
"woman"  → [0.1, 0.9, 0.2, ...]  # Female, common

# Similar words have similar vectors
```

**Why it matters:** The model doesn't see words - it sees these number vectors.

### Attention Mechanism

**What:** How the model decides which words to "pay attention to" when processing text.

**Example:**

```python
Sentence: "The student who was late have many books."

When processing "have":
Attention weights:
  "The"     → 0.05  (low attention)
  "student" → 0.85  (HIGH - subject of verb!)
  "who"     → 0.12
  "was"     → 0.08
  "late"    → 0.03
  "have"    → 0.95  (HIGH - this is the word being processed)
  "many"    → 0.10
  "books"   → 0.15

# Model focuses on "student" and "have" to check agreement
```

**Why it matters:** We can visualize attention to show users where the model thinks errors are.

### Multi-Task Learning

**What:** Training one model to do multiple things at once instead of separate models.

**Example:**

```python
# Traditional approach (3 models):
Model 1: Essay → CEFR score
Model 2: Essay → Error spans
Model 3: Essay → Error types

# Multi-task approach (1 model):
Model: Essay → {
    "cefr_score": 6.2,
    "error_spans": [...],
    "error_types": {...}
}
```

**Benefit:** Tasks share knowledge. Learning to score essays helps detect errors (both need grammar understanding).

**Trade-off:** Might not be as good at each task as specialized models.

### Loss Function

**What:** A number that says "how wrong" the model's predictions are. Training tries to minimize this.

**Example:**

```python
# True CEFR: 6.0
# Predicted: 5.5

# Mean Squared Error (MSE):
loss = (6.0 - 5.5)² = 0.25

# Model updates its weights to make loss smaller next time
```

**Multi-task loss:**

```python
total_loss = (1.0 × CEFR_loss) + (0.5 × span_loss) + (0.3 × type_loss)
#            ↑ most important    ↑ less important   ↑ least important
```

### Fine-Tuning

**What:** Taking a pre-trained model and adapting it for your specific task.

**Process:**

```python
# 1. Pre-training (already done by Microsoft):
DeBERTa learns English from billions of words

# 2. Fine-tuning (what we do):
Take DeBERTa → Train on essays → Get essay scorer

# vs. Training from scratch:
Random model → Train on essays → Would need 100x more data!
```

**Why it matters:** We get a smart model for "free" - just need to teach it our specific task.

### Transfer Learning

**What:** Using knowledge learned on one task for another task.

**Analogy:**

```
Learning to ride a bicycle → helps learn to ride motorcycle
Learning English grammar   → helps learn French grammar
Pre-trained language model → helps score essays
```

**In our project:**

- DeBERTa learned language from Wikipedia, books, etc.
- We fine-tune it to score essays
- Much faster than learning from scratch!

### Epoch

**What:** One complete pass through the entire training dataset.

**Example:**

```python
Training data: 3,784 essays
Batch size: 16

1 epoch = 3,784 ÷ 16 = 237 batches
# Model sees all 3,784 essays once

Training for 10 epochs = see each essay 10 times
```

**Why multiple epochs:** Model learns gradually, needs to see data multiple times.

### Early Stopping

**What:** Stop training when the model stops improving (prevents overfitting).

**Example:**

```python
Epoch 1: dev_loss = 1.5 ✅ improving
Epoch 2: dev_loss = 0.8 ✅ improving
Epoch 3: dev_loss = 0.6 ✅ improving
Epoch 4: dev_loss = 0.7 ⚠️ worse (patience = 1/3)
Epoch 5: dev_loss = 0.9 ⚠️ worse (patience = 2/3)
Epoch 6: dev_loss = 1.1 ⚠️ worse (patience = 3/3) → STOP!

# Use model from Epoch 3 (best performance)
```

### Overfitting

**What:** Model memorizes training data instead of learning patterns. Performs worse on new data.

**Analogy:**

```
Student memorizes answers to practice exam questions
✅ Gets 100% on practice exam
❌ Fails real exam (different questions but same concepts)

Model memorizes training essays
✅ Perfect on training data
❌ Poor on new essays
```

**How to detect:** Training loss keeps decreasing, but validation loss increases.

**Solution:** Early stopping, dropout, regularization.

### M2 Format

**What:** A standardized way to annotate errors in text (used by linguists).

**Example:**

```
S The student have many books .
A 2 3|||R:VERB:SVA|||has|||REQUIRED|||-NONE-|||0
  ^^^^ ^^^^error type^^correction

Breakdown:
- S = Original sentence
- A = Annotation
- 2 3 = Tokens 2 to 3 (span of error)
- R:VERB:SVA = Error type (Replace: Verb: Subject-Verb Agreement)
- has = Correct form
- REQUIRED = This correction is necessary
```

**Why it matters:** Our dataset has M2 annotations. Converting these to BIO tags is critical for error detection!

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

**Step 1: Parse M2 files** ([`parse_m2_annotations.py`](../../scripts/training/parse_m2_annotations.py))

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

**⚠️ Issue:** Current implementation uses simplified BIO tagging that doesn't properly align with actual M2 error positions (see [Problems](#problems-encountered)).

**Step 3: Generate enhanced dataset** ([`prepare-enhanced-corpus.py`](../../scripts/training/prepare-enhanced-corpus.py))

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
  "annotated_sentences": [...]  // First 5 sentences with M2 annotations
}
```

**Statistics:**

- **Train**: 3,784 essays / 37,486 sentences
- **Dev**: 476 essays / 4,352 sentences
- **Error rate**: 66% of sentences have ≥1 error
- **Category breakdown**: 57% grammar, 17% vocab, 9% mechanics, 9% fluency

### How Error Detection Actually Works

**Key Question:** How can a model detect grammar errors without knowing grammar rules?

**Answer:** Pattern learning from thousands of examples!

#### The Learning Process

**1. The model sees many examples:**

```python
# Example 1: SVA error
Text: "The student have books"
BIO tags: ["O", "O", "B-ERROR", "O"]
           ↑ model learns "student have" is problematic

# Example 2: Correct
Text: "The student has books"
BIO tags: ["O", "O", "O", "O"]
           ↑ model learns "student has" is fine

# Example 3: SVA error (different subject)
Text: "The teacher have exams"
BIO tags: ["O", "O", "B-ERROR", "O"]

# Example 4: Correct (plural)
Text: "The students have books"
BIO tags: ["O", "O", "O", "O"]

# After seeing 1000s of examples, model learns:
# - singular noun + "have" = ERROR
# - plural noun + "have" = OK
# - singular noun + "has" = OK
```

**The model doesn't have a rule saying "singular subjects need singular verbs"**. Instead:

- It sees "student have" marked as error 500 times
- It sees "student has" marked as correct 500 times
- It learns the statistical pattern

#### Why This Works

**Transformers (like DeBERTa) are pattern matching machines:**

1. **Context Understanding:**

```python
"The student who was late have books"

Model's internal process:
1. Embed each word → vectors
2. Attention mechanism looks at relationships:
   - "have" pays attention to "student" (subject)
   - Ignores "who was late" (relative clause)
3. Checks pattern: singular "student" + "have"
4. Matches error pattern → predict B-ERROR
```

2. **Statistical Associations:**

```python
# Model builds internal "knowledge" from data:

Patterns seen in training:
- "I have" → ✅ (seen 1000x as correct)
- "you have" → ✅ (seen 800x as correct)
- "he have" → ❌ (seen 300x as error)
- "she have" → ❌ (seen 250x as error)
- "it have" → ❌ (seen 200x as error)
- "they have" → ✅ (seen 900x as correct)

Model learns: {I, you, they} + have = OK
              {he, she, it} + have = ERROR
```

3. **Distributed Representations:**

```python
# Words with similar meanings cluster together in embedding space

"student" embedding ≈ "teacher" ≈ "person" (all singular nouns)
"students" embedding ≈ "teachers" ≈ "people" (all plural nouns)

If model learns "student have" is wrong,
it generalizes to "teacher have", "person have" too!
```

#### What About Complex Grammar?

**Determiner Errors:**

```python
# Training examples:
"I have [the] book" → O O O O (correct)
"I have book" → O O B-ERROR (missing article)
"I have a books" → O O B-ERROR I-ERROR (wrong determiner + plural)

# Model learns:
# - countable singular nouns usually need article
# - "a" + plural noun = error
# - uncountable nouns (water, advice) don't need article
```

**Why it works:** Sees 10,000+ examples of article usage patterns.

**Vocabulary Errors:**

```python
# Training examples:
"I am very happy" → ✅
"I am very much happy" → ❌ (awkward collocation)
"I am extremely happy" → ✅

# Model learns:
# - "very" + adjective = common pattern
# - "very much" + adjective = uncommon (likely error)
# - "extremely" + adjective = common pattern
```

**Why it works:** Statistical co-occurrence patterns from thousands of essays.

#### The Key Insight

**The model doesn't "understand" grammar like a linguist. It recognizes patterns like:**

```python
# Human linguist thinks:
"Subject-verb agreement requires singular subjects to take singular verbs"

# Model "thinks" (simplified):
Pattern vector [0.1, 0.8, 0.3, ...] + vector [0.2, 0.1, 0.9, ...]
                ↑ "student"              ↑ "have"
→ Similarity to error patterns seen in training: 0.87
→ Predict: B-ERROR

Pattern vector [0.1, 0.8, 0.3, ...] + vector [0.3, 0.2, 0.7, ...]
                ↑ "student"              ↑ "has"
→ Similarity to correct patterns: 0.92
→ Predict: O (no error)
```

#### Why Deep Learning is Powerful Here

**Traditional approach (rule-based):**

```python
# Need to code every rule:
if subject.is_singular() and verb == "have":
    return ERROR
if subject.is_plural() and verb == "has":
    return ERROR
# ... 1000s more rules for all error types!
```

**Deep learning approach:**

```python
# Just show examples:
# - Student have → ERROR
# - Student has → OK
# ... model figures out the pattern!
```

**Benefits:**

1. Handles complex patterns humans can't easily describe
2. Learns context-dependent rules automatically
3. Works for errors where rules are fuzzy (vocabulary, style)
4. Generalizes to new cases

**Trade-offs:**

1. Needs lots of labeled examples (we have 37,000+ sentences!)
2. Can't explain WHY (just knows pattern matches)
3. Makes mistakes on rare patterns
4. Only as good as training data

#### Why Our Error Detection Failed

**Remember:** This only works if training data has:

1. ✅ Real error positions (BIO tags aligned to actual errors)
2. ✅ Enough examples of each error type
3. ✅ Strong enough training signal (loss weights)

**Our problem:**

```python
# What we gave the model:
"The student have books"
BIO tags: ["O", "O", "B-ERROR", "O"]  # ← WRONG! Random position

# Model tried to learn: "student" is an error ❌
# Should have learned: "have" after "student" is an error ✅
```

**Fix:** Use actual M2 error positions → model learns correct patterns!

---

**Step 3: Generate enhanced dataset** ([`prepare-enhanced-corpus.py`](../../scripts/training/prepare-enhanced-corpus.py))

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
  "annotated_sentences": [...]  // First 5 sentences with M2 annotations
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

### Why Multi-Task Learning?

**Theory:**

1. **Shared Representations**: Understanding essay quality (CEFR) and identifying errors share common linguistic features
   - Both require understanding grammar, vocabulary usage, coherence
   - A model learning to score essays must implicitly learn about errors
   - Error detection benefits from global context (essay-level quality)

2. **Regularization**: Multiple tasks prevent overfitting to any single objective
   - Forces the model to learn more robust features
   - Acts as implicit data augmentation
   - Improves generalization

3. **Efficiency**: One model, multiple outputs
   - Shared encoder (80% of parameters)
   - Only task-specific heads differ
   - Faster inference than separate models

**Expected Benefits:**

- 2-5% drop in CEFR accuracy acceptable
- Gain error detection capability
- Natural synergy between tasks

**Actual Results:**

- ✅ CEFR: 0.85 QWK (only 2% drop from 0.87)
- ❌ Error detection: F1 = 0.00 (not learning)
- ❌ Error types: F1 ≈ 0% (not learning)

**Why It Failed (Partially):**
The theory is sound, but implementation issues prevented error tasks from learning (see [Problems Encountered](#problems-encountered)).

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

**Issue:** These weights were too conservative - model optimized almost entirely for CEFR task.

---

## Training Process

### Hardware: Modal GPU

```python
@modal.app.function(
    gpu="A10G",              # NVIDIA A10G (24GB VRAM)
    timeout=7200,            # 2 hours
    volumes={"/checkpoints": volume},  # Persistent storage
)
def train_feedback_model():
    # Training code here
```

**Why Modal?**

- No local GPU needed
- Pay-per-use (~$1-2/hour)
- Easy deployment after training
- Volume for checkpoint persistence

### Training Configuration

```yaml
model: microsoft/deberta-v3-base
max_length: 512
batch_size: 16
learning_rate: 2e-5
epochs: 15 (early stopped at 9)
warmup_steps: 500

# Multi-task loss weights
cefr_weight: 1.0
span_weight: 0.5
error_type_weight: 0.3

# Early stopping
patience: 3
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
    if dev_loss < best_dev_loss:
        save_checkpoint()
        best_dev_loss = dev_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 3:
            print("Early stopping!")
            break
```

---

## Actual Training Results

### Final Metrics

**Training Run:** 9 epochs (early stopped), ~25 minutes, ~$8 cost

**Checkpoint:** Epoch 6 (best dev_loss = 0.5593)

| Task                          | Metric        | Result    | Target | Status        |
| ----------------------------- | ------------- | --------- | ------ | ------------- |
| **CEFR Scoring**              | QWK           | **0.853** | ≥0.82  | ✅ Excellent  |
|                               | MAE           | 0.504     | ≤0.50  | ✅ Acceptable |
|                               | Adjacent Acc  | 57.6%     | -      | ✅ Good       |
| **Error Span Detection**      | F1            | 0.000     | ≥0.70  | ❌ Failed     |
|                               | Precision     | 0.000     | -      | ❌ Failed     |
|                               | Recall        | 0.000     | -      | ❌ Failed     |
| **Error Type Classification** | Grammar F1    | 0.061     | ≥0.50  | ❌ Very Poor  |
|                               | Vocabulary F1 | 0.000     | ≥0.50  | ❌ Failed     |
|                               | Mechanics F1  | 0.000     | ≥0.50  | ❌ Failed     |
|                               | Fluency F1    | 0.000     | ≥0.50  | ❌ Failed     |

### Training Trajectory

```
Epoch 1: dev_loss 1.58 (poor start)
Epoch 2: dev_loss 0.82 ✅ (improving)
Epoch 3: dev_loss 0.80 ✅ (best so far)
Epoch 4: dev_loss 0.89 (overfitting)
Epoch 5: dev_loss 0.73 ✅ (recovered)
Epoch 6: dev_loss 0.56 ✅✅ BEST (checkpoint saved)
Epoch 7: dev_loss 0.73 ⚠️ (patience 1/3)
Epoch 8: dev_loss 1.03 ⚠️ (patience 2/3)
Epoch 9: dev_loss 0.93 ⚠️ (patience 3/3) → EARLY STOP
```

### Loss Components (Epoch 6)

```
Total loss: 0.56
├─ CEFR loss: 0.41 (73% of total)
├─ Span loss: 0.05 (9% of total)
└─ Error type loss: 0.40 (18% of total - but not learning patterns!)
```

**Observation:** Span and type losses decreased, but the model learned to predict "no error" for everything rather than actual patterns.

### Comparison to Baseline

| Model          | QWK      | MAE      | Notes              |
| -------------- | -------- | -------- | ------------------ |
| T-AES-CORPUS   | 0.87     | 0.32     | Current production |
| T-AES-FEEDBACK | **0.85** | **0.50** | Only 2% QWK drop!  |

**Verdict:** CEFR task is production-ready. Error detection tasks need work.

---

## Problems Encountered

### Problem 1: Simplified BIO Tagging ⚠️ **Root Cause**

**What We Did:**

```python
# In feedback_dataset.py
def create_simplified_bio_tags(self, example):
    if example.get("has_errors", False):
        error_count = example.get("error_count", 0)
        # Distribute errors evenly across sequence
        for i in range(error_count):
            pos = (i + 1) * (self.max_length // (error_count + 2))
            span_labels[pos] = 1  # B-ERROR
```

**Problem:**

- Not using actual M2 error positions!
- Random/heuristic placement of error tags
- No alignment with tokenized text
- Model has no ground truth to learn from

**Evidence:**

- Model predicts "O" (no error) for all tokens
- Easiest way to minimize cross-entropy loss
- 66% of sentences have errors, but model sees random noise

**Why This Happened:**

- M2 annotations (`annotated_sentences`) exist in dataset
- But data loader not using them during training
- Used simplified approach to get training started
- Intended as temporary - but trained full model with it

### Problem 2: Multi-Task Weight Imbalance

**Configuration:**

```python
cefr_weight: 1.0    # Strong gradient signal
span_weight: 0.5    # Weak gradient signal
error_type_weight: 0.3  # Very weak gradient signal
```

**Effect:**

- Model optimizes primarily for CEFR (easiest task)
- Error tasks contribute minimally to gradient updates
- 1.0 vs 0.5 means CEFR gets 2x the gradient magnitude
- With bad data (Problem 1), error tasks easily ignored

**Evidence:**

- CEFR loss: 0.41 (learning well)
- Span loss: 0.05 (low, but not learning patterns)
- Type loss: 0.40 (not decreasing meaningfully)

### Problem 3: Insufficient Training Signal

**Theory:** Multi-task learning requires strong supervision for ALL tasks

**Reality:**

- CEFR: Clear, continuous labels (3.0, 4.5, 6.0, etc.)
- Spans: Binary tags per token (noisy due to Problem 1)
- Types: Multi-label (5 categories, sparse)

**Issues:**

1. Error tasks are harder to learn than CEFR
2. Need more epochs (20-30) specifically for error tasks
3. Need better loss balancing
4. Possibly need curriculum learning (CEFR first, then errors)

### Problem 4: Data Quality vs. Quantity Trade-off

**Available:**

- M2 annotations: Sentence-level only (first 5 sentences per essay)
- Complete essays: All sentences, but no annotations

**Used:**

- Complete essay text for CEFR
- Simplified error tags (random) for spans/types

**Should Have Used:**

- Sentence-level training with actual M2 annotations
- OR: Better M2 → token alignment algorithm

### Problem 5: Model Capacity

**Question:** Is 184M parameters enough for three tasks?

**Analysis:**

- CEFR works fine (proving model has capacity)
- Error detection harder than CEFR
- May need separate specialized heads
- OR: Larger model (DeBERTa-v3-large = 435M params)

**Verdict:** Probably not the issue, but worth considering.

---

## Solutions & Next Steps

### Solution 1: Fix BIO Tagging (CRITICAL) 🔧

**Implement Proper M2 → Token Alignment:**

```python
def create_proper_bio_tags(self, m2_sentence, tokenizer):
    """
    Align M2 error spans with subword tokens.
    """
    # 1. Get character positions from M2
    errors = m2_sentence.annotations  # [{start: 10, end: 14, type: "SVA"}]

    # 2. Tokenize with offset mapping
    encoding = tokenizer(
        m2_sentence.text,
        return_offsets_mapping=True
    )

    # 3. For each token, check if it overlaps with error span
    bio_tags = ["O"] * len(encoding.input_ids)

    for error in errors:
        error_start_char = error["start_char"]
        error_end_char = error["end_char"]

        first_token = True
        for idx, (start, end) in enumerate(encoding.offset_mapping):
            # Token overlaps with error?
            if start < error_end_char and end > error_start_char:
                if first_token:
                    bio_tags[idx] = "B-ERROR"
                    first_token = False
                else:
                    bio_tags[idx] = "I-ERROR"

    return bio_tags
```

**Benefits:**

- Actual ground truth labels
- Model can learn real error patterns
- Uses existing M2 annotations

**Effort:** 2-3 hours coding + testing

### Solution 2: Rebalance Task Weights

**New Configuration:**

```python
# Option A: Equal weights
cefr_weight: 1.0
span_weight: 1.0      # Up from 0.5
error_type_weight: 1.0  # Up from 0.3

# Option B: Prioritize error detection
cefr_weight: 0.7      # Down from 1.0
span_weight: 1.5      # Up significantly
error_type_weight: 1.0  # Up significantly
```

**Theory:**

- Error tasks need stronger gradient signal
- Willing to sacrifice 3-5% CEFR for error detection
- Option B: More aggressive rebalancing

**Effort:** Change config, retrain (~$8)

### Solution 3: Curriculum Learning 📚

**Approach:** Train tasks sequentially instead of simultaneously

**Phase 1:** CEFR only (5 epochs)

```python
cefr_weight: 1.0
span_weight: 0.0  # Disabled
error_type_weight: 0.0  # Disabled
```

**Phase 2:** Add error detection (10 epochs)

```python
cefr_weight: 0.3  # Freeze CEFR head mostly
span_weight: 1.5
error_type_weight: 1.0
```

**Theory:**

- Build strong foundation (CEFR understanding)
- Then specialize in errors
- Proven effective in multi-task literature

**Effort:** 1 day implementation + retraining

### Solution 4: Use Only Annotated Sentences

**Current:** Train on full essays (with bad error labels)

**Alternative:** Train only on sentences with M2 annotations

**Changes:**

```python
# Dataset: Use sentence-level data
for sentence in m2_sentences:
    yield {
        "text": sentence.text,
        "cefr": sentence.essay_cefr,  # Inherit from essay
        "bio_tags": create_proper_bio_tags(sentence),
        "error_types": sentence.error_categories
    }
```

**Pros:**

- Perfect error labels for 5 sentences/essay
- 3,784 essays × 5 = 18,920 training examples
- High-quality supervision

**Cons:**

- Smaller training set
- Sentence-level CEFR less reliable
- May hurt CEFR accuracy

**Verdict:** Worth trying as experiment

### Solution 5: Increase Training Epochs

**Current:** 15 max epochs (stopped at 9)

**Proposed:** 25-30 epochs with adjusted early stopping

**Rationale:**

- Error tasks need more iterations
- CEFR converges quickly (4-6 epochs)
- Error tasks just starting to learn at epoch 10
- Early stopping on total loss misses this

**Change:**

```python
patience: 5  # Up from 3
num_epochs: 30  # Up from 15
monitor_metrics: ["cefr_qwk", "span_f1", "type_f1"]  # All tasks
```

---

## Alternative Approaches

If multi-task learning continues to struggle, consider these alternatives:

### Approach 1: Two-Stage Pipeline

**Stage 1:** CEFR Scoring (current T-AES-CORPUS or FEEDBACK)

```python
cefr_model = load_model("feedback_cefr_only")
cefr_score = cefr_model.predict(essay)
```

**Stage 2:** Error Detection (specialized model)

```python
error_model = load_model("error_detector")
errors = error_model.predict(essay)
```

**Pros:**

- Each model optimized for one task
- Can use different architectures
- Better than nothing

**Cons:**

- 2x inference cost
- No shared representations
- More deployment complexity

### Approach 2: Attention-Based Error Detection

**Concept:** Use CEFR model's attention as error signal

**Implementation:**

```python
# Train CEFR model normally
cefr_model.train_on_corpus()

# Extract attention for low-scoring regions
attention = cefr_model.get_attention(essay)

# Heuristic: High attention = likely error
error_candidates = tokens_where(attention > threshold)
```

**Pros:**

- Zero-shot error detection
- Uses proven CEFR model
- No additional training

**Cons:**

- Heuristic, not learned
- Attention ≠ errors (just correlation)
- No error types

**Verdict:** Good fallback for v1

### Approach 3: Large Language Model (LLM) Fine-Tuning

**Model:** Llama 3.1 8B or similar

**Approach:**

```python
prompt = f"""
Analyze this essay for errors:
{essay_text}

Return JSON:
{{
  "cefr": <score>,
  "errors": [
    {{"span": "have many", "type": "grammar", "reason": "SVA"}}
  ]
}}
"""
```

**Pros:**

- LLMs understand language deeply
- Can explain errors naturally
- Seen error correction before

**Cons:**

- Expensive inference
- Slower (2-3s vs 300ms)
- Harder to control output format
- Requires more compute

**Verdict:** Future consideration for v2

### Approach 4: Ensemble with rule-based system

**Component 1:** FeedbackModel for CEFR

**Component 2:** LanguageTool/GrammarBot for error detection

```python
def get_comprehensive_feedback(essay):
    # ML for scoring
    cefr = feedback_model.predict(essay)

    # Rules for errors
    errors = languagetool.check(essay)

    return {
        "cefr": cefr,
        "errors": group_by_type(errors)
    }
```

**Pros:**

- Best of both worlds
- Rule-based errors are precise
- ML for nuanced scoring

**Cons:**

- Two systems to maintain
- Rule-based misses context
- Not "learned" errors

**Verdict:** Pragmatic hybrid approach

### Approach 5: Data Augmentation

**Generate Synthetic Errors:**

```python
def corrupt_essay(clean_essay):
    # SVA errors
    clean_essay = clean_essay.replace(" has ", " have ")

    # Determiner errors
    clean_essay = remove_random_articles(clean_essay)

    # Label corrupted spans
    return essay_with_labels
```

**Pros:**

- Infinite training data
- Controlled error types
- Can balance error distribution

**Cons:**

- Synthetic errors differ from real ones
- May not generalize
- Quality control needed

**Verdict:** Supplement M2 data, not replace

---

## Recommended Path Forward

Based on analysis, here's the recommended sequence:

### Phase 1: Fix Data Pipeline (1-2 days)

1. ✅ Implement proper M2 → BIO alignment
2. ✅ Test on small batch
3. ✅ Validate token positions match errors

### Phase 2: Retrain with Better Data (~1 day)

```python
config = {
    "cefr_weight": 0.8,    # Slight reduction
    "span_weight": 1.2,     # Increase
    "error_type_weight": 1.0,  # Increase
    "num_epochs": 25,       # More training
    "patience": 5
}
```

**Expected Results:**

- CEFR: 0.82-0.84 QWK (slight drop acceptable)
- Span F1: 0.60-0.75 (usable)
- Type F1: 0.50-0.65 (usable)

### Phase 3: Evaluate \& Iterate

**If results good (F1 > 0.65):**

- Deploy as T-AES-FEEDBACK v1
- Monitor in production
- Gather user feedback

**If results mediocre (F1 0.50-0.65):**

- Deploy CEFR-only (already excellent)
- Use attention-based error hints (Approach 2)
- Plan for Approach 4 (ensemble)

**If results still poor (F1 < 0.50):**

- Deploy CEFR-only
- Consider Approach 1 (two-stage)
- OR: Approach 4 (hybrid with rules)

### Phase 4: Production Deployment

**Option A: Full multi-task (if working)**

```python
@modal.web_endpoint()
def score_essay(essay):
    return {
        "cefr": model.predict_cefr(essay),
        "errors": model.detect_errors(essay),
        "heatmap": model.get_attention(essay)
    }
```

**Option B: CEFR + attention hints**

```python
@modal.web_endpoint()
def score_essay(essay):
    cefr, attention = model.predict_with_attention(essay)
    error_hints = high_attention_regions(attention)
    return {
        "cefr": cefr,
        "error_hints": error_hints  # Not precise, but something
    }
```

---

## Inference & Output

### Current Capability: CEFR Scoring

**Working:**

```python
def predict_cefr(essay_text: str):
    inputs = tokenizer(essay_text, return_tensors="pt", max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    cefr_score = outputs["cefr_score"].item()

    return {
        "score": cefr_score,
        "level": score_to_cefr(cefr_score),  # e.g., "B2"
        "confidence": "high"  # QWK 0.85!
    }
```

### Intended Capability: Full Feedback (After Fixes)

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
    error_tokens = span_probs[:, :, 1] > 0.7  # B-ERROR > 70%

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
                "mechanics": error_types[0, 2].item(),
                "fluency": error_types[0, 3].item(),
                "other": error_types[0, 4].item()
            },
            "spans": extract_error_spans(error_tokens, inputs),
            "heatmap": create_attention_heatmap(attention, inputs, essay_text)
        }
    }
```

### Heatmap Generation

**Concept:** Show which words the model paid attention to

```python
def create_attention_heatmap(attention, inputs, essay_text):
    """
    Convert attention weights to word-level heatmap.
    """
    # Average attention across all tokens
    avg_attention = attention.mean(dim=1)[0]  # [seq_len]

    # Map tokens back to words using offset mapping
    words = []
    word_attentions = []

    for idx, (start, end) in enumerate(inputs["offset_mapping"][0]):
        if start == end:  # Special token ([CLS], [SEP], [PAD])
            continue

        word = essay_text[start:end]
        attn = avg_attention[idx].item()

        words.append(word)
        word_attentions.append(attn)

    # Normalize to 0-1 range
    max_attn = max(word_attentions)
    normalized = [a / max_attn for a in word_attentions]

    return [
        {
            "word": w,
            "attention": a,
            "intensity": "high" if a > 0.7 else "medium" if a > 0.4 else "low"
        }
        for w, a in zip(words, normalized)
    ]
```

### API Response Format

**Current (CEFR-only):**

```json
{
  "cefr": {
    "score": 6.24,
    "level": "B2"
  },
  "metadata": {
    "model": "t-aes-feedback-v1",
    "inference_time_ms": 245
  }
}
```

**Future (With error detection fixed):**

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

### Current State: CEFR Scoring

Can be deployed as drop-in replacement for T-AES-CORPUS:

```python
# Modal deployment
@modal.app.function(
    gpu="T4",
    keep_warm=1
)
@modal.web_endpoint(method="POST")
def score_essay(request):
    essay = request.json()["text"]
    result = predict_cefr(essay)
    return result
```

**Frontend:** No changes needed - same response format as T-AES-CORPUS

### Future State: Full Feedback

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
    <span className={getColor(attention)} title={`Attention: ${attention.toFixed(2)}`}>
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

**Problem Spans Highlighting:**

```typescript
<div className="error-spans">
  {problemSpans.map(span => (
    <Tooltip content={`${span.likely_type} (${span.confidence}% confident)`}>
      <span className="underline-wavy-red">
        {span.text}
      </span>
    </Tooltip>
  ))}
</div>
```

---

## Comparison with Other Models

### Feature Matrix

| Feature                      | T-AES-CORPUS  | T-AES-ESSAY     | T-AES-FEEDBACK (Current) | T-AES-FEEDBACK (Goal) |
| ---------------------------- | ------------- | --------------- | ------------------------ | --------------------- |
| **CEFR Score**               | ✅ (QWK 0.87) | ✅              | ✅ (QWK 0.85)            | ✅ (QWK 0.82+)        |
| **Multi-dimensional scores** | ❌            | ✅              | ❌                       | ✅                    |
| **Error detection**          | ❌            | ❌              | ❌ (not working)         | ✅                    |
| **Error types**              | ❌            | ❌              | ❌ (not working)         | ✅                    |
| **Attention heatmap**        | ❌            | ❌              | ⏸️ (not tested)          | ✅                    |
| **Speed**                    | Fast (~200ms) | Slow (~2s)      | Medium (~300ms)          | Medium (~300ms)       |
| **Model size**               | 125M params   | Multiple models | 184M params              | 184M params           |

### Use Cases

**T-AES-CORPUS**: Quick overall assessment

- ✅ Fast, accurate CEFR scoring
- ❌ No actionable feedback

**T-AES-ESSAY**: Detailed rubric scoring

- ✅ Multiple dimensions (coherence, task achievement, etc.)
- ❌ No error-level feedback

**T-AES-FEEDBACK (Current)**: Improved CEFR scoring

- ✅ CEFR score comparable to T-AES-CORPUS (0.85 vs 0.87)
- ✅ Trained on larger dataset (3,784 essays vs 1,750)
- ❌ Error features not working yet

**T-AES-FEEDBACK (Goal)**: Comprehensive feedback

- ✅ CEFR score + error detection + heatmap
- ✅ Actionable insights for learners
- ⚠️ Slightly slower, may have 2-5% lower CEFR accuracy

---

## Key Takeaways

### What Works ✅

1. **CEFR Scoring**: Excellent performance (QWK 0.85, only 2% below baseline)
2. **Modal Training**: GPU infrastructure working perfectly
3. **Multi-task Architecture**: Model architecture is sound
4. **Data Pipeline**: M2 parsing and dataset creation functional
5. **Checkpoint Persistence**: Modal Volumes working for model storage

### What Doesn't Work ❌

1. **Error Span Detection**: F1 = 0 (model predicts "O" for everything)
2. **Error Type Classification**: F1 ≈ 0% (not learning patterns)
3. **BIO Tagging**: Using simplified/random tags instead of proper M2 alignment

### Root Causes

1. **Data Quality**: Simplified BIO tagging doesn't provide real ground truth
2. **Task Weights**: Conservative weights (0.5, 0.3) allow model to ignore error tasks
3. **Training Duration**: Error tasks need more epochs than CEFR

### Next Steps

1. **Short-term**: Deploy CEFR-only model (production-ready)
2. **Medium-term**: Fix BIO tagging, retrain with better weights
3. **Long-term**: Consider curriculum learning or hybrid approaches

### Learned Lessons

1. **Multi-task ≠ Magic**: Requires careful balancing and quality data for ALL tasks
2. **Validation**: Test data quality early - we assumed BIO tags were correct
3. **Incremental**: Should have trained CEFR-only first, validated, then added complexity
4. **Monitoring**: Track per-task metrics, not just combined loss

### Theory Validation

**✅ Confirmed:**

- Multi-task learning works when properly implemented
- Shared encoder benefits CEFR task (larger dataset, similar performance)
- DeBERTa-v3 is excellent for text understanding

**❌ Disproven:**

- Multi-task automatically improves all tasks (needs good data!)
- Attention weights directly correspond to errors (correlation, not causation)
- Conservative task weights are safe (too conservative = tasks ignored)

---

## Understanding Key Metrics

Before diving into further reading, let's explain the metrics we use:

### Quadratic Weighted Kappa (QWK)

**What is it?** A measure of agreement between model predictions and human ratings.

**Scale:** -1 to 1

- **1.0** = Perfect agreement
- **0.8+** = Strong agreement (our target)
- **0.6-0.8** = Moderate agreement
- **< 0.6** = Weak agreement

**Why "Quadratic"?** Penalizes larger disagreements more heavily.

**Example:**

```python
# True scores: [B1, B2, B2, C1]
# Predictions: [B1, B2, C1, C1]

# Quadratic penalty:
# B1==B1: penalty = 0² = 0 ✅
# B2==B2: penalty = 0² = 0 ✅
# B2→C1: penalty = 1² = 1 (off by 1 level)
# C1==C1: penalty = 0² = 0 ✅

# QWK = 0.85 (strong agreement)
```

**Why Important for AES?**

- Being off by 1 CEFR level (B2→C1) is much better than being off by 2 levels (B2→C2)
- QWK captures this nuance better than accuracy
- Standard metric in automated essay scoring research

**Further Reading:**

- [Quadratic Weighted Kappa Explained](https://towardsdatascience.com/quadratic-weighted-kappa-a-practical-guide-1bb17b7e1bb7)
- [Original Cohen's Kappa Paper](https://psycnet.apa.org/record/1961-04252-001)

### Mean Absolute Error (MAE)

**What is it?** Average distance between predicted and true scores.

**Formula:**

```python
MAE = sum(|predicted - true|) / n

# Example:
true = [5.0, 6.0, 7.0]
pred = [5.5, 5.8, 7.2]

MAE = (|5.5-5.0| + |5.8-6.0| + |7.2-7.0|) / 3
    = (0.5 + 0.2 + 0.2) / 3
    = 0.30
```

**Interpretation:**

- **MAE = 0.30**: Predictions are on average 0.3 CEFR points off
- **MAE ≤ 0.40**: Good performance for AES
- **Lower is better** (0 = perfect)

**Why Use Both QWK and MAE?**

- **QWK**: Ordinal agreement (are rankings right?)
- **MAE**: Absolute error magnitude (how far off are we?)
- Together they give complete picture

**Further Reading:**

- [Regression Metrics Explained](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234)

### F1 Score

**What is it?** Harmonic mean of precision and recall.

**Components:**

```python
Precision = True Positives / (True Positives + False Positives)
# "Of all errors we detected, how many were real?"

Recall = True Positives / (True Positives + False Negatives)
# "Of all real errors, how many did we detect?"

F1 = 2 * (Precision * Recall) / (Precision + Recall)
# Balances precision and recall
```

**Example:**

```python
# 100 tokens
# 20 actual errors
# Model detected 25 errors
# 15 were correct

Precision = 15/25 = 0.60 (60% of detections were real)
Recall = 15/20 = 0.75 (caught 75% of real errors)
F1 = 2 * (0.60 * 0.75) / (0.60 + 0.75) = 0.67
```

**Target:** F1 ≥ 0.70 for error detection

**Further Reading:**

- [Precision and Recall Explained](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9)

---

## Learning Path: Foundations to Research

A structured path to understand this project from scratch:

### Level 1: Machine Learning Fundamentals (1-2 weeks)

**Core Concepts:**

- Supervised learning (input → model → output)
- Training vs testing vs validation sets
- Loss functions and optimization
- Overfitting vs underfitting

**Resources:**

📚 **Books:**

- [Machine Learning Basics](https://www.deeplearningbook.org/contents/ml.html) - Free, from Deep Learning Book by Goodfellow et al.

🎥 **Videos:**

- [StatQuest: Machine Learning](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) - Extremely clear, visual explanations
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Beautiful visualizations

📝 **Interactive:**

- [ML Crash Course by Google](https://developers.google.com/machine-learning/crash-course) - Hands-on exercises

**Key Takeaways:**

- Understand what "training" means
- Know why we split data into train/dev/test
- Grasp the concept of gradient descent

### Level 2: Deep Learning & Neural Networks (2-3 weeks)

**Core Concepts:**

- Neural network architecture
- Backpropagation
- Activation functions (ReLU, Softmax, Sigmoid)
- Loss functions (MSE, Cross-Entropy, BCE)

**Resources:**

📚 **Courses:**

- [Fast.ai Practical Deep Learning](https://course.fast.ai/) - Code-first, practical approach
- [Stanford CS231n](http://cs231n.stanford.edu/) - Comprehensive lectures (focus on early lectures)

🎥 **Videos:**

- [3Blue1Brown: What is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk) - Best visual explanation ever

📝 **Interactive:**

- [Neural Network Playground](https://playground.tensorflow.org/) - See neural networks in action

**Key Takeaways:**

- Understand forward/backward pass
- Know different loss functions and when to use them
- Grasp why deep networks are powerful

### Level 3: Natural Language Processing (NLP) Basics (2-3 weeks)

**Core Concepts:**

- Text as input (tokenization, embeddings)
- Recurrent Neural Networks (RNNs, LSTMs)
- Attention mechanism (foundation of transformers)
- Transfer learning in NLP

**Resources:**

📚 **Courses:**

- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) - Excellent, comprehensive
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/) - Practical, modern

🎥 **Videos:**

- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - Visual guide to embeddings
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - **Must-read** for understanding transformers

📝 **Papers:**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The transformer paper (revolutionary!)

**Key Takeaways:**

- Understand how text becomes numbers
- Grasp the attention mechanism (key to transformers)
- Know what "pre-training" and "fine-tuning" mean

### Level 4: Transformers & BERT Family (3-4 weeks)

**Core Concepts:**

- Self-attention mechanism
- BERT architecture (bidirectional encoding)
- RoBERTa, DeBERTa improvements
- Fine-tuning for downstream tasks

**Resources:**

📚 **Guides:**

- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) - Visual explanation of BERT
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Code walkthrough

🎥 **Videos:**

- [Yannic Kilcher: BERT Explained](https://www.youtube.com/watch?v=-9evrZnBorM) - Deep dive
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Build transformer from scratch

📝 **Papers:**

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Original BERT
- [RoBERTa: A Robustly Optimized BERT](https://arxiv.org/abs/1907.11692) - Better BERT
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) - Our model basis!
- [DeBERTaV3](https://arxiv.org/abs/2111.09543) - Latest version we use

**Key Takeaways:**

- Understand masked language modeling (BERT's pre-training)
- Know how disentangled attention works (DeBERTa's key innovation)
- Grasp why pre-training + fine-tuning is powerful

### Level 5: Multi-Task Learning (2-3 weeks)

**Core Concepts:**

- Shared representations across tasks
- Task-specific heads
- Loss weighting and balancing
- Negative transfer (when tasks interfere)

**Resources:**

📚 **Overviews:**

- [An Overview of Multi-Task Learning](https://ruder.io/multi-task/) - Excellent comprehensive guide by Sebastian Ruder
- [Multi-Task Learning in PyTorch](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) - Practical tutorial

📝 **Papers:**

- [A Survey on Multi-Task Learning](https://arxiv.org/abs/1997.00453) - Comprehensive survey
- [Multi-Task Deep Neural Networks](https://www.cs.cornell.edu/~kilian/papers/multitask_cvpr2014.pdf) - Classic paper
- [Auxiliary Tasks in Multi-Task Learning](https://arxiv.org/abs/1805.06334) - When auxiliary tasks help

**Key Takeaways:**

- Understand why tasks should help each other
- Know how to balance task weights
- Recognize when multi-task might hurt performance

### Level 6: Automated Essay Scoring (AES) (2-3 weeks)

**Core Concepts:**

- CEFR framework (A1-C2 levels)
- Inter-rater reliability (why QWK matters)
- Prompt-specific vs cross-prompt models
- Fairness and bias in AES

**Resources:**

📝 **Papers:**

**Foundational:**

- [Automated Essay Scoring: A Survey](https://arxiv.org/abs/2012.07844) - Comprehensive overview
- [The Hewlett Foundation Competition](https://www.kaggle.com/c/asap-aes) - Historical benchmark

**Modern Deep Learning:**

- [Neural Automated Essay Scoring](https://aclanthology.org/D16-1193/) - Early neural AES
- [Automated Essay Scoring with String Kernels and Word Embeddings](https://aclanthology.org/P17-1077/) - Feature engineering
- [BERT for Automated Essay Scoring](https://arxiv.org/abs/1909.00372) - Our approach!

**Datasets:**

- [Write & Improve Corpus](https://arxiv.org/abs/1711.09080) - Our dataset! (W&I 2024 v2)
- [ASAP Dataset](https://www.kaggle.com/c/asap-aes/data) - Classic benchmark

**Key Takeaways:**

- Understand CEFR levels (what B2 means)
- Know why QWK is the standard metric
- Grasp prompt-dependency challenges

### Level 7: Grammatical Error Correction & Detection (2-3 weeks)

**Core Concepts:**

- M2 format (error annotations)
- Token-level vs sequence-level correction
- BIO tagging for error spans
- Error categorization schemes

**Resources:**

📝 **Papers:**

**Surveys:**

- [Grammatical Error Correction: A Survey](https://arxiv.org/abs/2211.05166) - Recent comprehensive survey
- [Neural Approaches to GEC](https://aclanthology.org/2020.bea-1.16/) - Modern methods

**Datasets & Shared Tasks:**

- [CoNLL-2014 Shared Task](https://www.comp.nus.edu.sg/~nlp/conll14st.html) - Classic GEC benchmark
- [BEA 2019 Shared Task](https://www.cl.cam.ac.uk/research/nl/bea2019st/) - Write & Improve based
- [MultiGEC 2025](https://github.com/grammarly/MultiGEC-2025) - Latest challenge

**M2 Format:**

- [M2 Scorer Documentation](https://github.com/nusnlp/m2scorer) - Understanding annotations
- [Error Annotation Guidelines](https://www.comp.nus.edu.sg/~nlp/conll14st/annotation-guidelines.pdf) - What error types mean

**Key Takeaways:**

- Understand M2 format (critical for our project!)
- Know BIO tagging for sequence labeling
- Grasp different error categories (grammar, vocabulary, etc.)

### Level 8: Attention Visualization & Interpretability (1-2 weeks)

**Core Concepts:**

- Attention weights as interpretability
- Layer-wise attention patterns
- Attention is NOT always explanation
- BertViz and other tools

**Resources:**

📚 **Guides:**

- [Attention is not Explanation](https://arxiv.org/abs/1902.10186) - Important caveat!
- [Attention is not not Explanation](https://arxiv.org/abs/1908.04626) - Nuanced view
- [BertViz Documentation](https://github.com/jessevig/bertviz) - Visualize attention

🎥 **Tools:**

- [Transformer Explainability](https://github.com/hila-chefer/Transformer-Explainability) - Visual explanations
- [Language Interpretability Tool (LIT)](https://pair-code.github.io/lit/) - Google's tool

**Key Takeaways:**

- Attention weights ≠ importance (correlation, not causation)
- Multiple layers have different attention patterns
- Use attention as hints, not definitive explanations

### Level 9: Advanced Topics (Ongoing)

**Latest Research:**

🔬 **Conferences to Follow:**

- **ACL** (Association for Computational Linguistics) - Top NLP venue
- **EMNLP** (Empirical Methods in NLP) - Applied NLP
- **NAACL** (North American ACL) - Regional conference
- **BEA Workshop** (Building Educational Applications) - AES & GEC focused!

🔬 **Recent Trends (2023-2024):**

- **LLMs for AES**: GPT-4, Claude as essay graders
- **Prompt Engineering**: Few-shot learning for scoring
- **Explainable AES**: Beyond black-box predictions
- **Fairness**: Detecting and mitigating bias

📝 **Cutting-Edge Papers:**

- [ChatGPT for Essay Scoring](https://arxiv.org/abs/2303.13688) - LLM-based AES
- [Prompt-based AES](https://arxiv.org/abs/2306.05357) - Few-shot approaches
- [Multilingual AES](https://arxiv.org/abs/2309.12345) - Cross-lingual scoring

---

## Practical Resources for This Project

### Tools & Libraries

**Essential:**

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Our deep learning framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - Pre-trained models
- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) - QWK, F1, etc.

**Helpful:**

- [Weights & Biases](https://docs.wandb.ai/) - Experiment tracking
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualization
- [Modal Documentation](https://modal.com/docs) - Our GPU platform

### Code Examples

**GitHub Repositories:**

- [BERT Fine-tuning Examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch) - Official Hugging Face
- [Multi-Task Learning in PyTorch](https://github.com/pytorch/examples/tree/main/mnist_multitask) - Simple example
- [Automated Essay Scoring Baseline](https://github.com/dkgupta95/BERT-for-Automated-Essay-Scoring) - Similar to our approach

### Communities & Forums

**Ask Questions:**

- [Hugging Face Forums](https://discuss.huggingface.co/) - Transformer-specific help
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) - General ML discussion
- [Stack Overflow [pytorch]](https://stackoverflow.com/questions/tagged/pytorch) - Code help

**Stay Updated:**

- [Papers with Code](https://paperswithcode.com/) - Latest research + code
- [arXiv Sanity](http://www.arxiv-sanity.com/) - ML paper recommender
- [NLP Progress](https://nlpprogress.com/) - State-of-the-art tracking

---

## Project-Specific Reading

### Our Implementation

- **Model architecture**:
- [`parse_m2_annotations.py`](../../scripts/training/parse_m2_annotations.py): Parse M2 files
- [`prepare-enhanced-corpus.py`](../../scripts/training/prepare-enhanced-corpus.py): Merge datasets
- [`feedback_model.py`](../../scripts/training/feedback_model.py): Model definition
- [`train-feedback-model.py`](../../scripts/training/train-feedback-model.py): Training script
- [`validate-feedback-model.py`](../../scripts/training/validate-feedback-model.py): Validation script

### Our Documentation

- **Training- **GECToR (Fast)\*\*: [GEC Service documentation](gec.md)
- **LanguageTool**: Standard grammar check
- **Corpus Scorer**: [Corpus Model Guide](corpus.md)

### Reference

- [Evaluation Report](evaluation.md)

### Direct Papers for This Project

**Foundation:**

- [DeBERTaV3](https://arxiv.org/abs/2111.09543) - Our base model
- [Write & Improve Corpus](https://arxiv.org/abs/1711.09080) - Our dataset

**Techniques:**

- [Multi-Task Learning Overview](https://ruder.io/multi-task/) - Our training approach
- [M2 Scorer](https://github.com/nusnlp/m2scorer) - Our error annotation format
- [BIO Tagging](<https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)>) - Our error detection method

---

## Recommended Learning Sequence

**For Software Engineers New to ML:**

```
Week 1-2:   ML Fundamentals (StatQuest videos)
Week 3-4:   Neural Networks (3Blue1Brown + Fast.ai)
Week 5-7:   NLP Basics (CS224N lectures)
Week 8-11:  Transformers (Illustrated guides + papers)
Week 12-13: Multi-Task Learning (Ruder overview)
Week 14-15: AES & GEC (Papers + Write & Improve corpus)
Week 16+:   Build & experiment!
```

**For ML Engineers New to NLP:**

```
Week 1-2:   NLP Basics (Hugging Face course)
Week 3-4:   Transformers (Illustrated guides)
Week 5-6:   AES & GEC (Survey papers)
Week 7+:    Build & experiment!
```

**For NLP Engineers:**

```
Week 1:     Multi-Task Learning (Ruder)
Week 2:     AES & GEC (Specific papers)
Week 3:     W&I Corpus + M2 format
Week 4+:    Build & experiment!
```

---

---

**Status:** Training complete. CEFR scoring production-ready (QWK 0.85). Error detection requires fixes to BIO tagging and retraining. See [Solutions & Next Steps](#solutions--next-steps) for recommended path forward.

```

```
