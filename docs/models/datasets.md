# Essay Scoring Datasets

This document catalogs the datasets used to train our essay scoring models, specifically focusing on **dimensional/multi-trait scoring** (Task Achievement, Coherence & Cohesion, Vocabulary, and Grammar).

## Use Case

We typically train in two stages:

1.  **Dimensional Pre-training**: Learning distinct traits (Grammar vs Vocabulary) using specific datasets.
2.  **CEFR Calibration**: Aligning scores to the CEFR standard using the Write & Improve corpus.

---

## Active Datasets (Currently Used)

These datasets are actively used in the training pipeline for **AES-DEBERTA**.

### 1. Write & Improve Corpus (W&I)

**Purpose**: CEFR Calibration & Main Baseline

| Attribute  | Value                                                       |
| ---------- | ----------------------------------------------------------- |
| **Size**   | ~3,800 training essays                                      |
| **Labels** | **Holistic CEFR** (A1-C2)                                   |
| **Source** | Cambridge English (Real learners)                           |
| **Usage**  | Used by `AES-CORPUS` (Main) and `AES-DEBERTA` (Stage 2 & 3) |

### 2. IELTS-WT2-LLaMa3-1k

**Purpose**: Dimensional Scoring Initialization

| Attribute  | Value                                                                                                      |
| ---------- | ---------------------------------------------------------------------------------------------------------- |
| **Size**   | ~1,000 essays                                                                                              |
| **Labels** | **4 Dimensions** (TA, CC, Vocab, Grammar) on 0-9 scale                                                     |
| **Source** | [HuggingFace (`123Harr/IELTS-WT2-LLaMa3-1k`)](https://huggingface.co/datasets/123Harr/IELTS-WT2-LLaMa3-1k) |
| **Usage**  | Used by `AES-DEBERTA` (Stage 1) to learn distinct traits.                                                  |

**Pros:**

- Exact match for our IELTS-style dimensions.
- High-quality synthetic/augmented data.

### 3. DREsS (Dataset for Rubric-based Essay Scoring)

**Purpose**: Dimensional Scoring Augmentation

| Attribute   | Value                                                     |
| ----------- | --------------------------------------------------------- |
| **Size**    | Varies                                                    |
| **Labels**  | **3 Dimensions** (Content, Organization, Language)        |
| **Mapping** | Content → TA, Organization → CC, Language → Vocab/Grammar |
| **Usage**   | Used by `AES-DEBERTA` (Stage 1)                           |

---

## Candidate Datasets (Future Expansion)

These datasets are high-quality candidates for improving model robustness in the future.

### 1. ASAP++ (Attribute-Specific Essay Grading)

**Best choice for US K-12 style scoring**

| Attribute       | Value                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------- |
| **Size**        | ~13,000 essays (8 prompts)                                                                   |
| **Dimensions**  | Content, Organization, Style, Conventions                                                    |
| **Score Range** | 0-6 per dimension                                                                            |
| **Download**    | [https://banuadrian.github.io/asap-plus-plus/](https://banuadrian.github.io/asap-plus-plus/) |
| **Essays**      | [Kaggle ASAP-AES](https://www.kaggle.com/c/asap-aes/data)                                    |
| **License**     | Research use                                                                                 |

**Pros:** True human rubric scores, widely used in research.
**Cons:** US K-12 prompts differ significantly from IELTS/EFL tasks.

### 2. ASAP 2.0 (2024)

**Newest, largest argumentative dataset**

| Attribute    | Value                                                                                                |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| **Size**     | ~24,000 essays                                                                                       |
| **Focus**    | Argumentative writing (more relevant to IELTS)                                                       |
| **Download** | [Kaggle ASAP 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2) |

**Pros:** Large size, argumentative focus.

### 3. ELLIPSE Dataset

**Best for EFL learner specificity**

| Attribute      | Value                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------- |
| **Size**       | ~6,500 essays                                                                                  |
| **Dimensions** | Cohesion, Syntax, Vocabulary, Phraseology, Grammar, Conventions                                |
| **Source**     | English Language Learners                                                                      |
| **Download**   | [Kaggle ELLIPSE](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) |

**Pros:** 6 dimensional scores, EFL student authors.

---

## Training Strategy

Our current `AES-DEBERTA` training strategy (`scripts/training/train-deberta-aes.py`) implements a **3-Stage Process**:

1.  **Stage 1: Dimensional Pre-Training**
    - Uses **IELTS-WT2** & **DREsS**
    - Objective: Learn to distinguish between Content, Organization, and Grammar.
    - Loss: MSE on dimensional scores (0-9 scale).

2.  **Stage 2: CEFR Calibration**
    - Uses **Write & Improve (W&I)**
    - Objective: Calibrate the overall score to the specific CEFR standards (A2-C2).
    - Loss: Ordinal Regression (CORN) on CEFR levels.

3.  **Stage 3: End-to-End Fine-Tuning**
    - Uses **All Datasets**
    - Objective: Balance dimensional accuracy with correct CEFR alignment.
    - Loss: Combined Weighted Loss.

## References

- [ASAP++ Project](https://banuadrian.github.io/asap-plus-plus/)
- [Kaggle ASAP-AES](https://www.kaggle.com/c/asap-aes)
- [Kaggle ELLIPSE](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)
- [DREsS Paper](https://arxiv.org/abs/2309.00000)
