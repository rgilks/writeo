# Essay Scoring Datasets

This document catalogs the datasets used to train our essay scoring models, specifically focusing on **dimensional/multi-trait scoring** (Task Achievement, Coherence & Cohesion, Vocabulary, and Grammar).

---

## ⚠️ Licensing Considerations

> [!IMPORTANT]
> Dataset licensing affects how trained models can be used. We document licensing status for all datasets used in training.

| Dataset                 | License Type          | Commercial Use | Notes                                                      |
| ----------------------- | --------------------- | -------------- | ---------------------------------------------------------- |
| **IELTS-WT2-LLaMa3-1k** | Unknown (HuggingFace) | ⚠️ Check       | No explicit license specified on HuggingFace               |
| **DREsS**               | Consent Form Required | ⚠️ Academic    | Requires signed consent form for access                    |
| **Write & Improve**     | Non-commercial only   | ❌ No          | Used only for calibration/validation, not primary training |
| **ASAP++**              | Research use          | ⚠️ Check       | Standard academic research license                         |
| **ELLIPSE**             | Kaggle Competition    | ⚠️ Check       | Competition rules apply                                    |

### Write & Improve Corpus Policy

The W&I Corpus has restrictive licensing:

- **Non-commercial use only** - prohibits commercial products/services
- **No derived items without approval** - models trained primarily on this data require CUP&A approval
- **No redistribution** - data cannot be shared publicly

**Our Approach**: AES-DEBERTA uses W&I only for **Stage 2 CEFR calibration** and **validation**. The primary model training uses other datasets (IELTS-WT2, DREsS). This may be defensible under research/educational use terms, but we do not redistribute the corpus or models primarily derived from it.

---

## Active Datasets (Currently Used)

These datasets are actively used in the training pipeline for **AES-DEBERTA**.

### 1. IELTS-WT2-LLaMa3-1k

**Purpose**: Dimensional Scoring Initialization (Primary Training Data)

| Attribute   | Value                                                                                                      |
| ----------- | ---------------------------------------------------------------------------------------------------------- |
| **Size**    | ~1,000 essays                                                                                              |
| **Labels**  | **4 Dimensions** (TA, CC, Vocab, Grammar) on 0-9 scale                                                     |
| **Source**  | [HuggingFace (`123Harr/IELTS-WT2-LLaMa3-1k`)](https://huggingface.co/datasets/123Harr/IELTS-WT2-LLaMa3-1k) |
| **License** | Unknown (no explicit license on HuggingFace)                                                               |
| **Usage**   | Used by `AES-DEBERTA` (Stage 1) to learn distinct traits.                                                  |

**Pros:**

- Exact match for our IELTS-style dimensions.
- High-quality synthetic/augmented data.

**Note:** License is listed as "unknown" on HuggingFace. Usage should be considered for research/educational purposes until clarified.

### 2. DREsS (Dataset for Rubric-based Essay Scoring)

**Purpose**: Dimensional Scoring Augmentation (Primary Training Data)

| Attribute   | Value                                                                  |
| ----------- | ---------------------------------------------------------------------- |
| **Size**    | ~48.9K samples (2.3K human-scored + 6.5K standardized + 40K synthetic) |
| **Labels**  | **3 Dimensions** (Content, Organization, Language) on 1-5 scale        |
| **Mapping** | Content → TA, Organization → CC, Language → Vocab/Grammar              |
| **Source**  | [Official Website](https://haneul-yoo.github.io/dress/)                |
| **License** | Consent form required for access (ACL 2025 paper)                      |
| **Usage**   | Used by `AES-DEBERTA` (Stage 1)                                        |

**Pros:**

- Large-scale rubric-based dataset.
- Expert-scored EFL essays.
- Published in ACL 2025.

**Note:** Access requires submitting a consent form. License terms are in the consent agreement.

### 3. Write & Improve Corpus (W&I)

**Purpose**: CEFR Calibration & Validation Only

| Attribute   | Value                                                |
| ----------- | ---------------------------------------------------- |
| **Size**    | ~3,800 training essays                               |
| **Labels**  | **Holistic CEFR** (A1-C2)                            |
| **Source**  | Cambridge University Press & Assessment              |
| **License** | Non-commercial, research use only, no redistribution |
| **Usage**   | Used by `AES-DEBERTA` (Stage 2 calibration only)     |

> [!WARNING]
> W&I license prohibits commercial use and requires approval for derived models. We use it only for calibration (aligning scores to CEFR scale) and validation (testing accuracy), not as primary training data.

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
    - Uses **IELTS-WT2** & **DREsS** (Primary training data)
    - Objective: Learn to distinguish between Content, Organization, and Grammar.
    - Loss: MSE on dimensional scores (0-9 scale).

2.  **Stage 2: CEFR Calibration**
    - Uses **Write & Improve (W&I)** (Calibration only)
    - Objective: Calibrate the overall score to the specific CEFR standards (A2-C2).
    - Loss: Ordinal Regression (CORN) on CEFR levels.
    - **Note**: W&I is used only for calibration, not primary training.

3.  **Stage 3: End-to-End Fine-Tuning**
    - Uses **IELTS-WT2 & DREsS** (Primary training data)
    - Objective: Balance dimensional accuracy with correct CEFR alignment.
    - Loss: Combined Weighted Loss.

## References

- [IELTS-WT2-LLaMa3-1k Dataset](https://huggingface.co/datasets/123Harr/IELTS-WT2-LLaMa3-1k)
- [DREsS Dataset](https://haneul-yoo.github.io/dress/) - ACL 2025
- [ASAP++ Project](https://banuadrian.github.io/asap-plus-plus/)
- [Kaggle ASAP-AES](https://www.kaggle.com/c/asap-aes)
- [Kaggle ELLIPSE](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)
