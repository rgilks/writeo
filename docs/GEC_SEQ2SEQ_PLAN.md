# T-GEC-SEQ2SEQ: Sequence-to-Sequence Error Correction Model

**Implementation Plan for Dedicated Grammatical Error Correction Model**

---

## Overview

**Goal:** Build a production-quality GEC model that corrects essays and provides detailed error spans with corrections.

**Approach:** Sequence-to-sequence (industry standard)

- Input: Grammatically incorrect text
- Output: Corrected text
- Post-process: Diff to extract error spans + corrections

**Model:** T5-base or BART-base (220M/140M params)

**Expected Performance:**

- F1: 0.60-0.75 (vs current 0.28)
- Precision: 0.70-0.85
- Recall: 0.50-0.65

---

## Architecture Comparison

### Current Approach (Token Classification)

```
Input: "I has three book"
Encoder → Token embeddings → BIO classifier
Output: [O, B-ERR, O, B-ERR]

Problems:
❌ Severe class imbalance (~95% O tags)
❌ No correction suggestions
❌ Competing with CEFR/error type tasks
❌ Low recall (0.192)
```

### New Approach (Seq2Seq)

```
Input: "I has three book"
Encoder-Decoder → Generated text
Output: "I have three books"

Then diff:
- "has" → "have" (grammar error)
- "book" → "books" (grammar error)

Benefits:
✅ Balanced training (all tokens matter)
✅ Gets corrections automatically
✅ Focused single-task model
✅ Industry-proven approach
```

---

## Phase 1: Data Preparation

### 1.1 Convert M2 to Seq2Seq Format

**Input:** M2 annotations

```
S This are a example .
A 1 2|||SVA|||is|||REQUIRED|||
A 3 4|||NOUN:NUM|||an|||REQUIRED|||
```

**Output:** Source-target pairs

```json
{
  "source": "This are a example.",
  "target": "This is an example."
}
```

**Script:** `scripts/training/prepare-gec-seq2seq.py`

```python
# Read M2 files
# Apply corrections to create target text
# Save as JSONL with source/target pairs
# Split: 80% train, 10% dev, 10% test
```

**Estimated examples:**

- Train: ~3,000 essays → ~50,000 sentence pairs
- Dev: ~400 essays → ~7,000 sentence pairs
- Test: ~400 essays → ~7,000 sentence pairs

### 1.2 Data Augmentation (Optional)

**Synthetic errors** to increase training data:

```python
# Inject common errors:
- Subject-verb agreement (I go → I goes)
- Verb tense (I go → I went)
- Articles (a apple → an apple)
- Plurals (book → books)

# Technique: Back-translation or rule-based
# Can 3x the dataset size
```

**Decision:** Start without, add if needed

---

## Phase 2: Model Selection

### Option A: T5-Base (Recommended)

**Model:** `t5-base` (220M params)
**Strengths:**

- Pre-trained on C4 (includes corrections)
- Text-to-text framework (natural fit)
- Strong performance on GEC benchmarks
- Good documentation

**Format:**

```
Input: "grammar: I has three book"
Output: "I have three books"
```

### Option B: BART-Base

**Model:** `facebook/bart-base` (140M params)
**Strengths:**

- Smaller/faster
- Denoising pre-training (similar task)
- Good for text generation

**Format:**

```
Input: "I has three book"
Output: "I have three books"
```

**Recommendation:** **T5-base** - better GEC results in literature

---

## Phase 3: Training Setup

### 3.1 Model Configuration

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-base"
max_source_length = 512  # Essay sentence
max_target_length = 512  # Corrected sentence

# Special tokens
prefix = "grammar: "  # T5 task prefix
```

### 3.2 Training Hyperparameters

**Initial config** (based on GEC literature):

```python
batch_size = 8  # Per device
grad_accumulation = 4  # Effective batch = 32
learning_rate = 5e-5
warmup_steps = 1000
num_epochs = 15
max_steps = ~25,000  # 15 epochs × ~1,700 steps/epoch

# Optimizer
optimizer = AdamW
weight_decay = 0.01
lr_scheduler = "linear"

# Regularization
dropout = 0.1
label_smoothing = 0.1  # Helps prevent overfitting
```

**Hardware:**

- GPU: A10G (24GB VRAM)
- Training time: ~3-4 hours
- Cost: ~$25-30

### 3.3 Training Script

**File:** `scripts/training/train-gec-seq2seq.py`

```python
import modal
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# Load data
train_data = load_dataset("train-gec-seq2seq.jsonl")
dev_data = load_dataset("dev-gec-seq2seq.jsonl")

# Tokenize
def preprocess(examples):
    inputs = [f"grammar: {text}" for text in examples["source"]]
    targets = examples["target"]

    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=512, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Training
training_args = Seq2SeqTrainingArguments(
    output_dir="/checkpoints/gec-seq2seq",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=True,
    generation_max_length=512,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=dev_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()
```

---

## Phase 4: Evaluation

### 4.1 Metrics

**1. ERRANT Scores** (official GEC metric)

```bash
# Install ERRANT
pip install errant

# Evaluate
errant_parallel -orig source.txt -cor hypothesis.txt -ref target.txt

# Outputs:
- Precision
- Recall
- F0.5 (standard for GEC, weights precision)
```

**2. GLEU Score**

```python
# Generalized Language Evaluation Understanding
# Smoother than BLEU for GEC
```

**Target Performance:**

- F0.5: ≥ 0.65 (industry standard)
- Precision: ≥ 0.75
- Recall: ≥ 0.50

### 4.2 Validation Script

**File:** `scripts/training/validate-gec-seq2seq.py`

```python
def generate_corrections(model, tokenizer, texts):
    inputs = tokenizer(
        [f"grammar: {text}" for text in texts],
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,  # Beam search for better quality
        early_stopping=True
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Compute metrics with ERRANT
from errant.annotator import Annotator
annotator = Annotator("en")

def evaluate(model, test_data):
    sources = [ex["source"] for ex in test_data]
    targets = [ex["target"] for ex in test_data]
    predictions = generate_corrections(model, tokenizer, sources)

    # ERRANT evaluation
    results = errant_evaluate(sources, predictions, targets, annotator)
    return results
```

---

## Phase 5: Error Span Extraction

### 5.1 Diff Algorithm

**Convert corrections to spans:**

```python
from difflib import SequenceMatcher

def extract_edits(source, target):
    """
    Extract edit operations between source and target.

    Returns list of edits with:
    - start/end positions
    - original text
    - correction
    - operation type (replace/insert/delete)
    """
    matcher = SequenceMatcher(None, source.split(), target.split())
    edits = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            edits.append({
                'type': 'replace',
                'start': i1,
                'end': i2,
                'original': ' '.join(source.split()[i1:i2]),
                'correction': ' '.join(target.split()[j1:j2]),
            })
        elif tag == 'delete':
            edits.append({
                'type': 'delete',
                'start': i1,
                'end': i2,
                'original': ' '.join(source.split()[i1:i2]),
                'correction': '',
            })
        elif tag == 'insert':
            edits.append({
                'type': 'insert',
                'start': i1,
                'end': i1,
                'original': '',
                'correction': ' '.join(target.split()[j1:j2]),
            })

    return edits

# Example
source = "I has three book"
target = "I have three books"
edits = extract_edits(source, target)
# [
#   {'type': 'replace', 'start': 1, 'end': 2, 'original': 'has', 'correction': 'have'},
#   {'type': 'replace', 'start': 3, 'end': 4, 'original': 'book', 'correction': 'books'}
# ]
```

### 5.2 Error Type Classification

**Map edits to error categories:**

```python
# Use ERRANT to classify error types
def classify_edit(source_tokens, target_tokens, edit, annotator):
    """
    Use ERRANT annotator to determine error type:
    - SVA (subject-verb agreement)
    - NOUN:NUM (noun number)
    - VERB:TENSE (verb tense)
    - etc.
    """
    errant_annot = annotator.annotate(source_tokens, target_tokens)
    # Map to our categories: grammar, vocabulary, mechanics, fluency
    return map_to_category(errant_annot)

def map_to_category(errant_type):
    grammar_types = ['SVA', 'VERB', 'NOUN', 'DET', 'PREP', 'ADJ', 'ADV']
    vocab_types = ['SPELL', 'WO']
    mechanics_types = ['PUNCT', 'ORTH']

    if any(t in errant_type for t in grammar_types):
        return 'grammar'
    elif any(t in errant_type for t in vocab_types):
        return 'vocabulary'
    elif any(t in errant_type for t in mechanics_types):
        return 'mechanics'
    else:
        return 'fluency'
```

---

## Phase 6: Modal Service Integration

### 6.1 Create GEC Service

**Directory:** `services/modal-gec/`

**Files:**

- `app.py` - Modal configuration
- `api.py` - FastAPI endpoint
- `model_loader.py` - Load T5 checkpoint
- `correction.py` - Generate corrections + extract edits

### 6.2 API Endpoint

```python
# services/modal-gec/api.py
from fastapi import FastAPI
from pydantic import BaseModel

class CorrectionRequest(BaseModel):
    text: str

class Edit(BaseModel):
    start: int
    end: int
    original: str
    correction: str
    type: str  # grammar/vocabulary/mechanics/fluency

class CorrectionResponse(BaseModel):
    original: str
    corrected: str
    edits: list[Edit]

@app.post("/correct")
async def correct_text(request: CorrectionRequest) -> CorrectionResponse:
    # Generate correction
    corrected = generate_correction(model, tokenizer, request.text)

    # Extract edits
    edits = extract_edits(request.text, corrected)

    # Classify error types
    for edit in edits:
        edit['type'] = classify_edit(request.text, corrected, edit, annotator)

    return CorrectionResponse(
        original=request.text,
        corrected=corrected,
        edits=edits
    )
```

### 6.3 Backend Integration

**Update ModalService:**

```typescript
// apps/api-worker/src/services/modal/types.ts
export interface ModalService {
  gradeEssay(request: ModalRequest): Promise<Response>;
  checkGrammar(text: string, language: string, answerId: string): Promise<Response>;
  scoreCorpus(text: string): Promise<Response>;
  scoreFeedback(text: string): Promise<Response>;
  correctGrammar(text: string): Promise<Response>; // NEW
}

// Response type
interface GECResponse {
  original: string;
  corrected: string;
  edits: Array<{
    start: number;
    end: number;
    original: string;
    correction: string;
    type: string;
  }>;
}
```

---

## Phase 7: Simplified T-AES-FEEDBACK

### 7.1 Remove Span Head

**Modify model architecture:**

```python
# scripts/training/feedback_model.py
class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_error_types=5):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)

        # CEFR head (keep)
        self.cefr_head = nn.Linear(hidden_size, 1)

        # Error type head (keep)
        self.error_type_head = nn.Linear(hidden_size, num_error_types)

        # REMOVE: self.span_head

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token

        return {
            "cefr_score": self.cefr_head(pooled).squeeze(-1),
            "error_type_logits": self.error_type_head(pooled),
            # REMOVE: "span_logits"
        }
```

### 7.2 Retrain (Quick)

```bash
# 5 epochs, focus on CEFR + error types
# Should improve error type classification
# Cost: ~$5-8
# Time: ~30 minutes
```

---

## Timeline & Costs

### Week 1: Data & Setup (8-12 hours)

- **Day 1-2:** Data preparation
  - Convert M2 to seq2seq format
  - Create train/dev/test splits
  - Validate data quality
- **Day 3:** Setup training
  - Configure Modal environment
  - Test T5 training pipeline
  - Set hyperparameters

**Cost:** $0 (data prep only)

### Week 2: Training & Evaluation (4-6 hours + compute)

- **Day 1:** Initial training run
  - Train T5-base for 15 epochs
  - Monitor training curves
  - Cost: ~$25-30
- **Day 2:** Evaluation & tuning
  - Run ERRANT evaluation
  - Adjust hyperparameters if needed
  - Possibly retrain (additional ~$15)
- **Day 3:** Validation
  - Test on held-out set
  - Manual review of corrections
  - Error analysis

**Cost:** $25-45 (training)

### Week 3: Integration (8-12 hours)

- **Day 1-2:** Modal service
  - Create GEC service
  - Implement correction + diff
  - Deploy and test
- **Day 2-3:** Backend integration
  - Update ModalService interface
  - Add to submission processor
  - Update frontend display
- **Day 3:** Testing
  - End-to-end testing
  - Performance validation
  - Documentation

**Cost:** $0 (integration only)

### Simplified Feedback Model (4 hours)

- Remove span head
- Quick retrain (5 epochs)
- Deploy updated version

**Cost:** ~$5-8

**Total:**

- **Time:** 20-30 hours of work over 2-3 weeks
- **Cost:** $30-53 total
- **Success probability:** 75-85%

---

## Success Criteria

### Minimum Viable Product (MVP)

- ✅ F0.5 ≥ 0.60
- ✅ Precision ≥ 0.70
- ✅ Generates valid corrections
- ✅ Extracts error spans with types
- ✅ Inference < 2 seconds per essay

### Production Ready

- ✅ F0.5 ≥ 0.65
- ✅ Precision ≥ 0.75
- ✅ Recall ≥ 0.50
- ✅ Inference < 1 second per essay
- ✅ Handles edge cases (very short/long text)

---

## Risk Mitigation

### Risk 1: Poor Performance (30% probability)

**Mitigation:**

- Start with pre-trained T5 (already good at text-to-text)
- Use proven hyperparameters from literature
- Fallback: Use BART or try ByT5 (character-level)

### Risk 2: Insufficient Data (20% probability)

**Mitigation:**

- Augment with synthetic errors
- Use additional public datasets (FCE, CoNLL-2014)
- Start with what we have, iterate

### Risk 3: Slow Inference (15% probability)

**Mitigation:**

- Use beam_size=1 for faster generation
- Consider distillation (T5-small)
- Batch processing

---

## Next Steps

**Immediate (Today):**

1. Review this plan
2. Confirm we want seq2seq approach
3. Decide on T5-base vs BART-base

**This Week:**

1. Create `prepare-gec-seq2seq.py` script
2. Convert M2 data to source-target pairs
3. Validate data quality (manual review of 50 examples)

**Next Week:**

1. Create training script
2. Run initial 15-epoch training
3. Evaluate with ERRANT

Want me to start with the data preparation script?
