# GEC Service Documentation

**Grammatical Error Correction Services**

---

## Overview

Writeo provides two GEC services that run in parallel:

| Service     | Model        | Speed  | Quality | Assessor ID   |
| ----------- | ------------ | ------ | ------- | ------------- |
| **Seq2Seq** | Flan-T5      | 12-16s | High    | `GEC-SEQ2SEQ` |
| **GECToR**  | RoBERTa-base | 1-2s   | Good    | `GEC-GECTOR`  |

Both services are enabled by default in `assessors.json` and their results appear as separate assessors in the response.

---

## Seq2Seq GEC (`modal-gec`)

**Approach:** Sequence-to-Sequence (Rewrite entire sentences)

### Architecture

```
Input: "I has three book"
   ↓
[Seq2Seq Model (Flan-T5)]
   ↓
Output: "I have three books"
   ↓
[Diff Algorithm (ERRANT/difflib)]
   ↓
Edits: "has" → "have", "book" → "books"
```

### Configuration

- **Model:** `google/flan-t5-base` (220M params, fine-tuned on GEC data)
- **GPU:** A10G
- **Keep-Warm:** 30s
- **Location:** `services/modal-gec/`

### Pros/Cons

✅ High precision and recall  
✅ Can make complex structural changes  
❌ Slow (~12-16s per request)  
❌ More expensive (A10G GPU)

---

## GECToR (`modal-gector`)

**Approach:** Token-level tagging (Tag, Not Rewrite)

### Architecture

```
Input: "I has three book"
   ↓
[GECToR Model (RoBERTa encoder)]
   ↓
Tags: [KEEP, REPLACE:have, KEEP, REPLACE:books]
   ↓
[Apply Tags]
   ↓
Output: "I have three books"
```

### Configuration

- **Model:** `gotutiyan/gector-roberta-base-5k`
- **GPU:** T4 (cheaper)
- **Keep-Warm:** 30s
- **Location:** `services/modal-gector/`

### Pros/Cons

✅ Very fast (~1-2s per request, ~10x faster)  
✅ Cheaper (T4 GPU)  
✅ Good for simple errors  
❌ May struggle with complex structural errors  
❌ Edit extraction can cascade on insertions

---

## API Reference

Both services use the same request/response format:

**Seq2Seq:** `POST /gec_endpoint`  
**GECToR:** `POST /gector_endpoint`

**Request:**

```json
{
  "text": "I has three book."
}
```

**Response:**

```json
{
  "original": "I has three book.",
  "corrected": "I have three books.",
  "edits": [
    {
      "start": 2,
      "end": 5,
      "original": "has",
      "correction": "have",
      "operation": "replace",
      "category": "grammar"
    },
    {
      "start": 12,
      "end": 16,
      "original": "book",
      "correction": "books",
      "operation": "replace",
      "category": "grammar"
    }
  ]
}
```

---

## Backend Integration

1. **Config:** Set via `assessors.json` (`gecSeq2seq`, `gecGector`)
2. **Execution:** Both called in parallel via `services.ts`
3. **Merging:** Results merged into `AssessmentResults` under their assessor IDs
4. **Display:** Edits converted to `LanguageToolError` format for heatmap display

---

## Deployment

```bash
# Deploy Seq2Seq
cd services/modal-gec && modal deploy main.py

# Deploy GECToR
cd services/modal-gector && modal deploy main.py
```
