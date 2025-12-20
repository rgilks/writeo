# Essay Scoring Models

This service supports multiple essay scoring models for comparison and selection.

## Available Models

### 1. `AES-DEBERTA` (Default) ✅ Production

- **Model**: `microsoft/deberta-v3-large` (Fine-tuned)
- **Type**: Multi-head regression + Ordinal regression
- **Output**:
  - **Overall**: 0-9 band scale
  - **Dimensions**: Task Achievement, Coherence & Cohesion, Vocabulary, Grammar
  - **CEFR**: A2-C2 level (Predicted by dedicated head)
- **Status**: ✅ Primary Assessor
- **Best For**: General purpose essay scoring, detailed dimensional feedback.
- **Documentation**: [AES-DEBERTA](deberta.md)

### 2. `AES-FEEDBACK` 🟡 Experimental

- **Model**: DeBERTa-v3-base (Multi-task)
- **Type**: Multi-task learning
- **Output**: CEFR Score + Error Detection (WIP)
- **Status**: 🟡 CEFR scoring works well, error detection is experimental.
- **Best For**: Secondary CEFR signal.
- **Documentation**: [AES-FEEDBACK](feedback.md)

### 3. `AES-ESSAY` (Legacy) 🔴 Deprecated

- **Model**: `KevSun/Engessay_grading_ML`
- **Output**: 6 dimensional scores
- **Status**: 🔴 Deprecated (Replaced by `AES-DEBERTA`)
- **Note**: Previously known as `engessay`.

---

## GEC Services

Grammar correction is handled by dedicated services running in parallel with scoring.

- **GEC-SEQ2SEQ**: Flan-T5 based rewriting (High quality, slower).
- **GEC-GECTOR**: RoBERTa tagging (Fast, good quality).

See [GEC Documentation](gec.md) for details.

---

## Usage

### Default Model

```bash
POST /grade
# Uses AES-DEBERTA by default
```

### Specify Model

```bash
POST /grade?model_key=deberta  # AES-DEBERTA
POST /grade?model_key=essay    # AES-ESSAY (Legacy)
```

## Comparisons

See [Evaluation Report](evaluation.md) for a detailed performance comparison of these models.
