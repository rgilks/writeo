# AES-CORPUS: Training Guide for Software Engineers

A practical guide to understanding how the AES-CORPUS assessor was trained and works.

## Table of Contents

- [What is AES-CORPUS?](#what-is-aes-corpus)
- [Training Data](#training-data)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [How It Makes Predictions](#how-it-makes-predictions)
- [Performance](#performance)
- [Deployment](#deployment)

---

## What is AES-CORPUS?

**AES-CORPUS** is an AI model that automatically scores English essays and assigns CEFR levels (A2, B1, B2, C1, C2). It was trained specifically on the **Write & Improve corpus** - a dataset of real student essays with expert CEFR labels.

**Key Facts:**

- **Input**: Essay text (up to 512 tokens/words)
- **Output**: Numeric score (2.0-8.5) + CEFR level (A2-C2)
- **Primary metric**: QWK 0.87 (excellent, near human-level agreement)
- **Use case**: Provides fast, accurate overall proficiency assessment

---

## Training Data

### Source: Write & Improve Corpus

The [Write & Improve corpus](https://www.cambridge.org/elt/blog/2018/08/07/write-improve-free-tool-improve-english-writing/) is a public dataset from Cambridge English containing:

- **~5,000 essays** from real English learners
- **Expert CEFR labels** (A1 through C2)
- **Authentic writing** from controlled assessment tasks
- **Pre-split** into train/dev/test sets to prevent data leakage

### Data Preparation

3. **Converts** CEFR levels to numeric scores
4. **Splits** data based on original corpus metadata:
   - **Train**: 3,784 essays (80%) - used to train the model
   - **Dev**: 476 essays (10%) - used to tune hyperparameters
   - **Test**: 481 essays (10%) - held out for final validation

**Critical**: The test set is never seen during training, ensuring unbiased evaluation.

```bash
# Prepare the corpus data
python scripts/training/prepare-corpus.py
```

---

## Model Architecture

### Base Model: RoBERTa

AES-CORPUS is built on **RoBERTa** (Robustly Optimized BERT), a transformer-based language model:

- **Pre-trained** on billions of words of English text
- **Understands** grammar, vocabulary, coherence, and writing style
- **General-purpose** - we fine-tune it for essay scoring

Think of RoBERTa as a "language understanding engine" that already knows English extremely well. We're just teaching it to map that understanding to CEFR scores.

### Architecture Overview

```
Essay Text
    ↓
Tokenizer (splits into ~500 word pieces)
    ↓
RoBERTa Encoder (12 transformer layers)
    ↓
[CLS] token representation (captures essay meaning)
    ↓
Regression Head (2 fully-connected layers)
    ↓
Score (2.0 - 8.5) → CEFR level (A2-C2)
```

**Key Components:**

1. **Tokenizer**: Breaks text into ~500 sub-words (handles all vocabulary)
2. **Encoder**: 12 layers of transformers that build rich text representation
3. **Regression Head**: Simple neural network that maps to score
4. **Output**: Single number representing essay proficiency

---

## Training Process

### 1. Setup

The training happens on **Modal's cloud GPU** to handle the computational load:

```python
# Modal setup (from train-overall-score.py)
@app.function(
    gpu="A10G",  # NVIDIA A10G GPU
    timeout=3600,  # 1 hour max
    secrets=[modal.Secret.from_name("hf-token")],  # Hugging Face access
)
```

### 2. Training Configuration

Key settings (from [`config.py`](../../scripts/training/config.py)):

```python
# Model
model_name: "roberta-base"  # Pre-trained RoBERTa

# Training
batch_size: 16              # Process 16 essays at once
learning_rate: 2e-5         # How fast the model learns
num_epochs: 15              # Complete passes through data
max_length: 512             # Max essay length in tokens

# Loss function
loss: "MSE"                 # Mean Squared Error (for regression)
```

### 3. Training Loop

For each of 15 epochs:

```python
for epoch in range(15):
    for batch of essays in training_data:
        # 1. Forward pass: model predicts scores
        predictions = model(essay_texts)

        # 2. Compute loss: how wrong are we?
        loss = mse_loss(predictions, true_scores)

        # 3. Backward pass: update model weights
        loss.backward()
        optimizer.step()

    # Evaluate on dev set
    dev_qwk = evaluate(dev_data)

    # Save best model
    if dev_qwk > best_qwk:
        save_model()
```

### 4. Loss Function

**Mean Squared Error (MSE)** measures prediction error:

```
MSE = average((predicted_score - true_score)²)
```

- Lower is better (perfect = 0)
- Penalizes large errors more than small ones
- Example: If true score is 6.0:
  - Predicting 6.5 → error = 0.25
  - Predicting 7.0 → error = 1.00 (4x worse!)

### 5. Early Stopping

Training stops when:

- **QWK stops improving** for 3 epochs (prevents overfitting)
- **15 epochs completed** (max limit)
- **Best model saved** based on dev set QWK

**Final result**: Model typically converges after ~10-12 epochs.

---

## How It Makes Predictions

### Step-by-Step Inference

When scoring a new essay:

```python
# 1. Tokenize the essay
tokens = tokenizer(essay_text, max_length=512, truncation=True)
# Output: [101, 2003, 1037, ... 102] (token IDs)

# 2. Pass through RoBERTa encoder
outputs = model(**tokens)
# Output: Rich representation of essay meaning

# 3. Extract [CLS] token (overall representation)
cls_output = outputs.last_hidden_state[:, 0, :]

# 4. Pass through regression head
score = regression_head(cls_output)
# Output: 6.24 (for example)

# 5. Map to CEFR level
cefr = score_to_cefr(score)  # 6.24 → B2
```

### Score to CEFR Mapping

The model outputs a continuous score (2.0-8.5) that we map to CEFR:

| Score Range | CEFR Level | Proficiency         |
| ----------- | ---------- | ------------------- |
| 8.25 - 8.5  | C2         | Mastery             |
| 7.75 - 8.25 | C1+        | Advanced+           |
| 7.0 - 7.75  | C1         | Advanced            |
| 6.25 - 7.0  | B2+        | Upper-Intermediate+ |
| 5.5 - 6.25  | B2         | Upper-Intermediate  |
| 4.75 - 5.5  | B1+        | Intermediate+       |
| 4.0 - 4.75  | B1         | Intermediate        |
| 3.25 - 4.0  | A2+        | Elementary+         |
| 2.75 - 3.25 | A2         | Elementary          |

These thresholds align with **IELTS band scores** for educational compatibility.

---

## Performance

### Validation Results

Tested on 481 held-out essays:

| Metric                | Value    | Interpretation                |
| --------------------- | -------- | ----------------------------- |
| **QWK**               | **0.87** | Excellent (near human-level)  |
| **MAE**               | 0.32     | Average error of 0.32 points  |
| **RMSE**              | 0.39     | Low prediction variance       |
| **Correlation**       | 0.93     | Strong agreement with experts |
| **Exact Accuracy**    | 60%      | Correct CEFR level            |
| **Adjacent Accuracy** | **100%** | Always within ±1 level        |

### What Does QWK 0.87 Mean?

**Quadratic Weighted Kappa (QWK)** measures agreement with expert scorers:

- **1.0** = Perfect agreement
- **0.75-1.0** = Excellent (this model!)
- **0.60-0.75** = Good
- **0.40-0.60** = Moderate
- **<0.40** = Needs improvement

**0.87 is approaching human-level agreement** between expert raters.

### Per-CEFR Level Performance

| CEFR | Sample Count | MAE  | Quality   |
| ---- | ------------ | ---- | --------- |
| A2   | 45           | 0.28 | Excellent |
| B1   | 189          | 0.31 | Excellent |
| B2   | 156          | 0.34 | Excellent |
| C1   | 78           | 0.35 | Excellent |
| C2   | 13           | 0.29 | Excellent |

**Consistent performance across all proficiency levels.**

---

## Deployment

### Modal Service

The model is deployed as a FastAPI service on Modal:

**URL**: `https://rob-gilks--writeo-corpus-fastapi-app.modal.run`

**Endpoint**: `POST /score`

**Request**:

```json
{
  "text": "Essay text here...",
  "max_length": 512
}
```

**Response**:

```json
{
  "score": 6.24,
  "cefr_level": "B2",
  "model": "roberta-base-cefr"
}
```

### Usage in Application

The model is called via- **[`ModalClient`](../../apps/api-worker/src/services/modal/client.ts)**: API interfacescript

```typescript
async scoreCorpus(text: string): Promise<Response> {
  return fetch(`${this.config.modal.corpusUrl}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, max_length: 512 }),
  });
}
```

**Dev Mode Integration**: When dev mode is enabled, the corpus score appears alongside AES-ESSAY for comparison.

---

## Key Takeaways

1. **Purpose**: Fast, accurate CEFR proficiency assessment
2. **Training**: Fine-tuned RoBERTa on 3,784 expert-labeled essays
3. **Performance**: QWK 0.87 (excellent), 100% adjacent accuracy
4. **Output**: Single overall score + CEFR level
5. **Deployment**: Modal cloud service with <1s response time

**Why it works well:**

- Large pre-trained model (understands English deeply)
- Quality training data (expert CEFR labels)
- Proper validation (held-out test set)
- Right architecture (transformers excel at text understanding)

---

## Educational Resources

### Understanding the Fundamentals

**For Software Engineers New to ML:**

#### Transformers & Language Models

- 📚 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Best visual guide to understand transformers (30 min read)
- 📚 [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) - How BERT works, applies to RoBERTa (20 min read)
- 🎥 [But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M) - 3Blue1Brown's excellent intro to language models (27 min)
- 📖 [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter1/1) - Free interactive course on transformers

#### Fine-tuning Explained

- 📚 [Transfer Learning Guide](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) - Concept applies to text too (15 min)
- 📚 [Fine-tuning Transformers](https://huggingface.co/docs/transformers/training) - Official Hugging Face docs
- 🎥 [Fine-tuning BERT](https://www.youtube.com/watch?v=x66kkDnbzi4) - Practical walkthrough (15 min)

#### CEFR Framework

- 📖 [CEFR Overview](https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions) - Official Council of Europe guide
- 📚 [CEFR for Teachers](https://www.cambridgeenglish.org/exams-and-tests/cefr/) - Cambridge English explanation
- 📊 [CEFR vs IELTS Mapping](https://www.ielts.org/for-organisations/ielts-scoring-in-detail) - How they align

#### Machine Learning Metrics

- 📚 [Kappa Score Explained](https://towardsdatascience.com/interpretation-of-kappa-values-2acd1ca7b18f) - Understanding agreement metrics (10 min)
- 📚 [Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) - MAE, RMSE, etc.
- 📊 [Confusion Matrix Guide](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) - Classification metrics (8 min)

### Technical References

**Code & Implementation:**

- **Training**: [`train-overall-score.py`](../../scripts/training/train-overall-score.py)
- **Evaluation**: [`evaluate-model.py`](../../scripts/training/evaluate-model.py)
- **Validation**: [`validate-assessors.py`](../../scripts/training/validate-assessors.py)
- **Configuration**: [`config.py`](../../scripts/training/config.py)

**Research Papers:**

- 📄 [RoBERTa: Robustly Optimized BERT](https://arxiv.org/abs/1907.11692) - Model architecture paper
- 📄 [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Original BERT paper
- 📄 [Write & Improve Dataset](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data) - Training data source

**Tools & Libraries:**

- 🛠️ [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Library we use
- 🛠️ [PyTorch](https://pytorch.org/tutorials/) - Deep learning framework
- 🛠️ [Modal](https://modal.com/docs/guide) - GPU cloud platform
- 🛠️ [scikit-learn](https://scikit-learn.org/stable/user_guide.html) - Metrics library

### Learning Path Recommendation

**If you're new to ML**, follow this order:

1. **Start**: Watch [But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M) (27 min)
2. **Read**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (30 min)
3. **Understand**: [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) (20 min)
4. **Apply**: Read our [training script](../../scripts/training/train-overall-score.py) with new context
5. **Metrics**: [Kappa Score Explained](https://towardsdatascience.com/interpretation-of-kappa-values-2acd1ca7b18f) (10 min)

**Total time**: ~2 hours to solid understanding ✅

---

## Quick Reference

**Model**: RoBERTa-base fine-tuned for CEFR scoring  
**Training data**: 3,784 essays from Write & Improve corpus  
**Performance**: QWK 0.87 (excellent), 100% adjacent accuracy  
**Deployment**: Modal FastAPI service  
**Response time**: < 1 second  
**Use case**: Fast, accurate overall CEFR assessment
