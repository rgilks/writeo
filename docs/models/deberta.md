# AES-DEBERTA Model Documentation

**Model ID:** `AES-DEBERTA`  
**Model Base:** `deberta-v3-large`  
**Task:** Automated Essay Scoring (Multi-dimensional)  
**Status:** Production (Default Assessor)  
**Last Updated:** December 07, 2025

---

## 1. Overview

`AES-DEBERTA` is the successor to the legacy `AES-ESSAY` model. It leverages the Microsoft DeBERTa V3 Large architecture, fine-tuned to predict multiple dimensions of essay quality simultaneously. Unlike the previous model which was trained primarily on IELTS data, this model incorporates a broader range of datasets to improve generalization.

### Capabilities

The model provides scores (0-9 scale) for the following dimensions:

- **Task Achievement (TA)**: How well the response addresses the prompt.
- **Coherence & Cohesion (CC)**: The logical flow and organization of ideas.
- **Vocabulary (Vocab)**: Range and accuracy of lexical choices.
- **Grammar**: Range and accuracy of grammatical structures.
- **Overall**: A holistic score aggregating the above dimensions.
- **CEFR Level**: An estimated CEFR level (A2-C2) derived from the Overall score.

---

## 2. Architecture

- **Base Model:** `microsoft/deberta-v3-large`
- **Head Architecture:** Multi-head regression.
  - A shared encoder outputs a pooled representation.
  - Five separate regression heads (linear layers) project the pooled output to the 5 target scores (TA, CC, Vocab, Grammar, Overall).
- **Max Sequence Length:** 1024 tokens (captures full essay context).
- **Input:** Concatenated prompt and essay text.

---

## 3. Training Data & Strategy

### Datasets

The model was trained on a composite dataset to ensure robustness:

1.  **IELTS-WT2**: Standard IELTS writing task 2 essays (primary source).
2.  **DREsS (Standard)**: Domain-specific essay dataset.
3.  **W&I (Write & Improve)**: Cambridge English dataset for CEFR calibration.

### Data Split Strategy

- **Training Set (80%)**: Used for model optimization.
- **Validation Set (10%)**: Used for hyperparameter tuning and early stopping.
- **Test Set (10%)**: Held-out data for internal evaluation.
- **DREsS_New**: A completely separate dataset used as a strict held-out test to evaluating generalization to unseen domains.

### Training Procedure (3-Stage)

1.  **Stage 1: Main Training (IELTS + DREsS)**
    - Objective: Minimize Mean Squared Error (MSE) on dimensional scores.
    - Strategy: Fine-tune the full model. This teaches the model the core scoring logic.
2.  **Stage 2: CEFR Calibration (W&I)**
    - Objective: Optimize for CEFR level alignment using the high-quality W&I dataset.
    - Strategy: Low learning rate fine-tuning to align internal representations with CEFR standards without forgetting dimensional scoring.
3.  **Stage 3: Fine-Grained Polish (Combined)**
    - Objective: Balance all objectives.
    - Strategy: Train on the combined dataset with a weighted loss function to ensure the model performs well across all dimensions and dataset types.

---

## 4. Performance

The model was evaluated against the legacy `AES-ESSAY` on the difficult `DREsS_New` held-out test set.

| Metric (MAE)   | AES-ESSAY (Legacy) | AES-DEBERTA (New) | Improvement |
| :------------- | :----------------- | :---------------- | :---------- |
| **Overall**    | 1.57               | **1.30**          | **+17.2%**  |
| **TA**         | 1.58               | **1.32**          | **+16.5%**  |
| **Coherence**  | 1.56               | **1.28**          | **+17.9%**  |
| **Vocabulary** | 1.56               | **1.30**          | **+16.7%**  |
| **Grammar**    | 1.58               | **1.30**          | **+17.7%**  |

_Note: Lower MAE (Mean Absolute Error) is better._

**Conclusion:** `AES-DEBERTA` consistently outperforms the legacy model by a significant margin (~0.27 points MAE) on unseen data, indicating much stronger generalization capabilities.

---

## 5. Integration

### API Usage

The model is deployed as a Modal service and accessed via the API Worker.

**Service ID:** `AES-DEBERTA`  
**Endpoint:** `score_deberta(text: str)`

**Request:**

```json
{
  "text": "The essay content..."
}
```

**Response:**

```json
{
  "type": "grader",
  "overall": 6.5,
  "label": "B2",
  "dimensions": {
    "TA": 6.0,
    "CC": 6.5,
    "Vocab": 7.0,
    "Grammar": 6.0,
    "Overall": 6.5
  },
  "metadata": {
    "model": "deberta-v3-large",
    "inference_time_ms": 145
  }
}
```

### Configuration

Enabled in `apps/api-worker/src/config/assessors.json`:

```json
"scoring": {
  "deberta": true,
  "essay": false  // Legacy model disabled
}
```
