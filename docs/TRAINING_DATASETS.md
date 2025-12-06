# Essay Scoring Datasets for Multi-Dimensional Training

This document catalogs publicly available datasets that could be used to train essay scoring models with **dimensional/multi-trait scoring** (TA, CC, Vocab, Grammar, etc.) rather than just holistic CEFR levels.

## Current Limitation

Our **T-AES-CORPUS** model (RoBERTa) was trained on the Write & Improve corpus which only has **holistic CEFR labels** (no dimensional breakdown). This is why we can't show consistent dimensional scores alongside the corpus score.

---

## High-Priority Datasets

### 1. ASAP++ (Attribute-Specific Essay Grading)

**Best choice for multi-dimensional scoring**

| Attribute       | Value                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------- |
| **Size**        | ~13,000 essays (8 prompts)                                                                   |
| **Dimensions**  | Content, Organization, Style, Conventions                                                    |
| **Score Range** | 0-6 per dimension                                                                            |
| **Download**    | [https://banuadrian.github.io/asap-plus-plus/](https://banuadrian.github.io/asap-plus-plus/) |
| **Essays**      | [Kaggle ASAP-AES](https://www.kaggle.com/c/asap-aes/data)                                    |
| **License**     | Research use                                                                                 |

**Pros:**

- Has true dimensional rubric scores from human graders
- Well-documented, widely used in research
- Easy to download

**Cons:**

- US K-12 essays (not adult EFL/IELTS style)
- Fixed prompts (8 total)

---

### 2. ASAP 2.0 (2024)

**Newest, largest dataset**

| Attribute    | Value                                                                                                |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| **Size**     | ~24,000 essays                                                                                       |
| **Focus**    | Argumentative writing (more relevant to IELTS)                                                       |
| **Download** | [Kaggle ASAP 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2) |
| **Year**     | 2024                                                                                                 |

**Pros:**

- Large dataset
- Argumentative essays (closer to IELTS Task 2)
- Recent competition with active community

---

### 3. ELLIPSE Dataset

**Best for EFL learners**

| Attribute      | Value                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------- |
| **Size**       | ~6,500 essays                                                                                  |
| **Dimensions** | Cohesion, Syntax, Vocabulary, Phraseology, Grammar, Conventions                                |
| **Source**     | English Language Learners                                                                      |
| **Download**   | [Kaggle ELLIPSE](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) |

**Pros:**

- 6 dimensional scores!
- Written by ELL students (matches our target users)
- Vocabulary and Grammar as separate dimensions

**Cons:**

- US middle/high school students

---

### 4. DREsS (Dataset for Rubric-based Essay Scoring)

**Academic EFL focus**

| Attribute       | Value                                      |
| --------------- | ------------------------------------------ |
| **Size**        | Varies by source                           |
| **Dimensions**  | Content, Organization, Language            |
| **Score Range** | 1-5 (0.5 increments)                       |
| **Paper**       | [ACL Anthology](https://aclanthology.org/) |

**Pros:**

- Combines multiple sources including ASAP++
- Standardized rubric format
- EFL-focused

---

## IELTS-Specific Datasets

### 5. IELTS Writing Scored Essays (Kaggle)

| Attribute        | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| **Dimensions**   | Task Achievement, Coherence & Cohesion, Lexical Resource, Grammatical Range |
| **Score Range**  | 0-9 band scores                                                             |
| **CEFR Mapping** | Direct (IELTS → CEFR is well-established)                                   |
| **Download**     | [Kaggle Search](https://www.kaggle.com/search?q=ielts+writing+scored)       |

**Pros:**

- Exact match for IELTS dimensions (TA, CC, Vocab, Grammar)
- Band scores map directly to CEFR

**Cons:**

- Various quality datasets on Kaggle
- May need validation

---

## Other Relevant Datasets

| Dataset     | Size           | Dimensions    | Notes                                   |
| ----------- | -------------- | ------------- | --------------------------------------- |
| **ICLE**    | ~6,000         | Holistic      | International Corpus of Learner English |
| **TOEFL11** | 12,100         | Holistic + L1 | Native language identification focus    |
| **FCE**     | 1,244          | Holistic      | Cambridge First Certificate essays      |
| **LOCNESS** | ~300,000 words | None          | Native English control corpus           |

---

## Recommended Training Strategy

### Option A: ASAP++ for Quick Multi-Dimensional Model

1. Download ASAP++ attribute scores
2. Download essays from Kaggle ASAP-AES
3. Train multi-output RoBERTa model
4. Map scores to IELTS-style dimensions

### Option B: ELLIPSE for EFL-Specific Training

1. Download ELLIPSE dataset from Kaggle
2. Use 6 trait scores as targets
3. Train multi-task model similar to T-AES-FEEDBACK

### Option C: Combine Multiple Datasets

1. ASAP++ for content/organization
2. ELLIPSE for grammar/vocabulary
3. Write & Improve for CEFR calibration
4. Multi-task learning across all

---

## Dimension Mapping

| Our Dimensions       | ASAP++          | ELLIPSE                 | IELTS             |
| -------------------- | --------------- | ----------------------- | ----------------- |
| Task Achievement     | Content         | -                       | Task Response     |
| Coherence & Cohesion | Organization    | Cohesion                | CC                |
| Vocabulary           | Style (partial) | Vocabulary, Phraseology | Lexical Resource  |
| Grammar              | Conventions     | Grammar, Syntax         | Grammatical Range |

---

## Next Steps

1. **Evaluate ASAP++ or ELLIPSE** as training data
2. **Train multi-output model** with dimension-specific heads
3. **Validate** against Write & Improve corpus (CEFR alignment)
4. **Replace T-AES-ESSAY** with new dimensional model

## References

- [ASAP++ Project](https://banuadrian.github.io/asap-plus-plus/)
- [Kaggle ASAP-AES](https://www.kaggle.com/c/asap-aes)
- [Kaggle ELLIPSE](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)
- [DREsS Paper](https://arxiv.org/abs/2309.00000)
