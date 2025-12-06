# GEC Service Documentation

**Service for Grammatical Error Correction using Seq2Seq Models**

---

## Overview

The GEC Service provides high-quality grammatical error correction for essays. Unlike the previous token-classification approach, this service uses a Sequence-to-Sequence (Seq2Seq) architecture to rewrite sentences with corrections.

**Key Features:**

- **Model:** `google/flan-t5-base` (220M params) fine-tuned on GEC data.
- **Approach:** Generates corrected text, then computes diffs to extract precise error spans.
- **Performance:** Higher precision and recall compared to token-tagging models.
- **Isolation:** Runs as a standalone separate Microservice on Modal.

---

## Architecture

### Model Flow

```
Input: "I has three book"
   ↓
[Seq2Seq Model (Flan-T5)]
   ↓
Output: "I have three books"
   ↓
[Diff Algorithm (ERRANT/difflib)]
   ↓
Edits:
- "has" -> "have" (Subject-Verb Agreement)
- "book" -> "books" (Noun Number)
```

### Infrastructure

- **Platform:** Modal
- **GPU:** A10G (for training), CPU/GPU (for inference)
- **Endpoint:** FastAPI (`/gec_endpoint`)

---

## Implementation Details

### 1. Data Preparation

- **Script:** `scripts/training/prepare-gec-seq2seq.py`
- **Source:** M2 formatted annotation files.
- **Output:** JSONL files (`train.jsonl`, `dev.jsonl`) containing `source` (incorrect) and `target` (correct) pairs.

### 2. Training

- **Script:** `scripts/training/train-gec-seq2seq.py`
- **Framework:** HuggingFace Transformers (`Seq2SeqTrainer`).
- **Location:** Runs on Modal, checkpoints saved to `writeo-gec-models` Volume.

### 3. Service Logic

- **Location:** `services/modal_gec/`
- **Correction Logic:** `correction.py` uses `difflib` (fallback) or `ERRANT` (if available) to align original and corrected sentences and extract spans.
- **API:** Returns a list of edits, each with `start`, `end`, `original`, `correction`, and `type`.

---

## API Reference

**Endpoint:** `POST /gec_endpoint`

**Request:**

```json
{
  "text": "I has a error."
}
```

**Response:**

```json
{
  "original": "I has a error.",
  "corrected": "I have an error.",
  "edits": [
    {
      "start": 2,
      "end": 5,
      "original": "has",
      "correction": "have",
      "type": "grammar"
    },
    {
      "start": 6,
      "end": 7,
      "original": "a",
      "correction": "an",
      "type": "grammar"
    }
  ]
}
```

---

## Backend Integration

The API Worker calls this service in parallel with other assessment services.

1. **Config:** `MODAL_GEC_URL` in `.dev.vars` / environment.
2. **Execution:** `modalService.correctGrammar(text)` called in `services.ts`.
3. **Merging:** Results are merged into the final `AssessmentResults` under the assessor ID `T-GEC-SEQ2SEQ`.
4. **Display:** Edits are converted to `LanguageToolError` format and appear in the heatmap alongside LT/LLM errors.
