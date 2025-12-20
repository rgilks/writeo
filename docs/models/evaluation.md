# Writeo Assessor Evaluation Report

## 1. Executive Summary

This report evaluates the performance, consistency, and utility of the various assessment models currently deployed in the Writeo API pipeline. The evaluation was conducted against the live API using a random sample of 20 essays from the `Write & Improve` test corpus, covering CEFR levels A2 to C2.

**Key Findings:**

- **Scoring Accuracy:** `AES-DEBERTA` is the highest-performing scoring model, achieving a correlation of **0.97** with human labels and the lowest Mean Absolute Error (0.38).
- **Consensus:** There is a strong alignment between `AES-CORPUS` and `AES-FEEDBACK` (DeBERTa), whereas `AES-ESSAY` consistently underestimates proficiency.
- **Feedback Redundancy:** The pipeline currently runs three separate Grammar Error Correction (GEC) engines (`LT`, `LLM`, `Seq2Seq`), leading to overlapping and potentially overwhelming feedback for the user.
- **Recommendations:** Consolidate scoring to use `AES-DEBERTA` as the primary source, and unify GEC feedback by prioritizing `GEC-SEQ2SEQ` for inline edits while using `GEC-LLM` for deeper explanations.

---

## 2. Scoring Models Evaluation

We evaluated three primary grading assessors against human-labeled CEFR scores converted to a 0-9 scale.

### Performance Metrics

| Assessor         | Model Type           | MAE (Error) | Bias  | Correlation (r) | Status              |
| ---------------- | -------------------- | ----------- | ----- | --------------- | ------------------- |
| **AES-CORPUS**   | RoBERTa (Regression) | **0.41**    | +0.18 | **0.96**        | 🟢 **Recommended**  |
| **AES-FEEDBACK** | DeBERTa (Multi-task) | 0.58        | +0.31 | 0.89            | 🟡 Secondary Signal |
| **AES-DEBERTA**  | DeBERTa (Regression) | **0.38**    | +0.12 | **0.97**        | 🟢 **Primary**      |
| **AES-ESSAY**    | Standard ML          | 0.75        | -0.55 | 0.86            | 🔴 Deprecated       |

### Analysis

- **AES-CORPUS**: Demonstrates high reliability. The slight positive bias (+0.18) is negligible and often preferred in learning contexts to encourage users. It tracks human scores linearly across the difficulty spectrum.
- **AES-FEEDBACK**: Slightly overestimates performance (+0.31). It is useful as a corroborating signal but less precise than the corpus-trained specific model.
- **AES-DEBERTA**: The new primary model. It outperforms all legacy models with the lowest MAE (0.38 vs 0.41) and highest correlation (0.97). It provides granular dimensional scores (TA, CC, Vocab, Grammar) with high reliability.
- **AES-ESSAY**: Consistently harsh grading (negative bias of -0.55). It frequently scores B2 essays as B1/A2+. This model is now deprecated in favor of `AES-DEBERTA`.

---

## 3. Feedback Models Evaluation

The pipeline produces qualitative feedback from multiple sources. We analyzed the utility and distinctness of each.

### Grammar & Error Correction (GEC)

| Assessor        | Type                      | Strengths                                                                                         | Weaknesses                                                              |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **GEC-SEQ2SEQ** | Seq2Seq (Flan-T5)         | Provides precise, diff-based edits (insert/replace/delete). Best for "Accept Change" UI features. | Can sometimes miss subtle context-dependent errors.                     |
| **GEC-LLM**     | LLM (Llama-3)             | Good at explaining _why_ something is wrong ("Verb tense error"). Catches stylistic issues.       | Output is verbose and computationally expensive. Overlaps with Seq2Seq. |
| **GEC-LT**      | Rule-based (LanguageTool) | Excellent for mechanics (typos, spacing, punctuation). Deterministic and fast.                    | False positives on creative/informal writing. Rigid.                    |

**Observation:** Running all three results in duplicate notifications for the same error (e.g., a missing comma might be flagged by all three).

### Thematic & Structural Feedback

- **AI-FEEDBACK**: Uses an LLM to provide holistic "Strengths" and "Improvements".
  - _Quality_: High. Provides actionable, high-level advice (e.g., "Develop your ideas with specific examples").
  - _Utility_: Critical for showing the user _how_ to improve, not just _what_ they got wrong.
- **TEACHER-FEEDBACK**: Simulates a teacher's voice.
  - _Analysis_: Often redundant with `AI-FEEDBACK`. It provides a shorter, more personal summary but content-wise is very similar.
  - _Recommendation_: Merge with `AI-FEEDBACK` into a single "Coach" persona response to reduce token usage and UI clutter.

### Relevance Checking

- **RELEVANCE-CHECK**: Embedding-based cosine similarity.
  - _Performance_: Correctly identifies off-topic essays (scores < 0.6) vs relevant ones (scores > 0.8).
  - _Utility_: Essential safety check to prevent high scores on irrelevant input.

---

## 4. Latency & Reliability

- **Reliability**: The API demonstrated robust handling of requests, though initial validation strictness (requiring `submission.0.answers.0.text`) caused some client-side friction.
- **Latency**:
  - **AES-DEBERTA**: Significant optimization achieved by baking model weights into the image and using FP16. Latency reduced from ~14s to **~0.4s** per request.
  - **Legacy LLM Assessors**: The heavy reliance on LLMs (Llama-3-70b) for multiple assessors (`GEC-LLM`, `AI-FEEDBACK`, `TEACHER-FEEDBACK`) remains the primary bottleneck for total request duration.

---

## 5. Strategic Recommendations

### 1. Unified Score Strategy

- **Primary Score**: Use `AES-DEBERTA`.
- **Confidence Interval**: Use `AES-FEEDBACK` to define a confidence range. If the two models diverge by > 1.0 points, flag the essay for human review or show a wider estimated band to the user.

### 2. De-duplicate GEC

- **Pipeline Change**:
  1. Run `GEC-SEQ2SEQ` to generate the "suggested edits" layer.
  2. Run `GEC-LT` for "mechanics" layer (spellcheck).
  3. **Disable** `GEC-LLM` for standard checking, OR run it only if the other two find nothing. Alternatively, use it only to _explain_ complex errors found by Seq2Seq.

### 3. Consolidate Qualitative Feedback

- Remove `TEACHER-FEEDBACK` as a separate entity.
- Enhance `AI-FEEDBACK` to include a "Teacher's Note" section if the "personal touch" is desired. This saves one distinct LLM call/parsing step.

### 4. Optimize Response Payload

- The current JSON response is deeply nested and verbose. Flatten the `assessorResults` into a synthesized `report` object for the frontend, doing the merging on the worker side to save bandwidth and frontend processing logic.
