# Essay Scoring Models

This service supports multiple essay scoring models for comparison and selection.

## Available Models

### 1. `engessay` (Default) ✅ Recommended for Production

- **Model**: `KevSun/Engessay_grading_ML`
- **Citation**: Sun, K., & Wang, R. (2024). Automatic Essay Multi-dimensional Scoring with Fine-tuning and Multiple Regression. _ArXiv_. https://arxiv.org/abs/2406.01198
- **Type**: RoBERTa-based sequence classification
- **Output**: 6 scores (1-5 scale) mapped to assessment dimensions
- **Dimensions**: cohesion, syntax, vocabulary, phraseology, grammar, conventions
- **Status**: ✅ Loaded and working - has complete weights (`pytorch_model.bin`)
- **Performance**: Provides differentiated scores across essay quality levels (5.0-9.0 band scale)
- **Mapping to assessment dimensions**:
  - cohesion → CC (Coherence & Cohesion)
  - syntax → Grammar
  - vocabulary → Vocab
  - phraseology → Vocab
  - grammar → Grammar
  - conventions → TA (Task Achievement)
- **Best For**: Production use - most reliable and feature-rich
- **Suitability**: Well-suited for academic argumentative writing practice. The model provides strong coverage of Coherence & Cohesion, Lexical Resource (vocabulary), and Grammatical Range & Accuracy. Task Achievement is assessed separately via LLM feedback, ensuring comprehensive evaluation across all key writing dimensions.

### 2. `distilbert`

- **Model**: `Michau96/distilbert-base-uncased-essay_scoring`
- **Type**: DistilBERT-based sequence classification
- **Output**: Single score normalized to 0-9 band scale
- **Status**: ✅ Loaded (uses base DistilBERT tokenizer)
- **Performance**: Produces similar scores to Engessay (may need calibration)
- **Best For**: Comparison/testing, lighter model option

### 3. `corpus-roberta`

- **Model**: Custom trained RoBERTa-base model
- **Type**: RoBERTa-based regression
- **Output**: Single overall score (2-9 band scale)
- **Training**: Fine-tuned on Write and Improve corpus (2024 v2) with ~5,050 essays
- **Status**: ⚠️ Requires training (see training scripts)
- **Performance**: Trained specifically on Write and Improve data with human CEFR labels
- **Assessor ID**: `T-AES-CORPUS`
- **Best For**: Comparison with engessay model, potentially better alignment with Write and Improve corpus annotations
- **Note**: Model must be trained before use. See `scripts/training/` for training instructions.

### 4. `fallback`

- **Type**: Heuristic-based scoring
- **Output**: Word-count based estimation
- **Status**: ✅ Always available
- **Performance**: Basic but reliable fallback
- **Best For**: Emergency fallback when models fail

## Usage

### Default Model

```bash
POST /grade
# Uses DEFAULT_MODEL (currently "engessay")
```

### Specify Model

```bash
POST /grade?model_key=engessay
POST /grade?model_key=distilbert
POST /grade?model_key=corpus-roberta
POST /grade?model_key=fallback
```

### List Available Models

```bash
GET /models
# Returns status of all models
```

### Compare Models

```bash
POST /grade/compare
# Scores the same essay with all available models
```

## Model Selection

The default model can be changed via:

1. Environment variable: `MODEL_NAME=engessay` (in Modal)
2. Query parameter: `?model_key=engessay`
3. Code: Update `DEFAULT_MODEL` in `services/modal-essay/app.py`

## Test Results

Models were tested across essays of varying quality:

| Essay Quality | Words | Engessay Overall | DistilBERT Overall |
| ------------- | ----- | ---------------- | ------------------ |
| Very Short    | 4     | 5.0              | 5.0                |
| Short Basic   | 9     | 5.5              | 5.5                |
| Medium Avg    | 29    | 6.0              | 6.0                |
| Long Good     | 44    | 7.0              | 7.0                |
| Excellent     | 58    | 7.5              | 7.5                |

## Training Custom Models

The `corpus-roberta` model can be trained using the Write and Improve corpus:

1. **Prepare data**: Run `scripts/training/prepare-corpus.py` to prepare training data
2. **Train model**: Run `scripts/training/train-overall-score.py` with `--test-run` for quick validation, or `--full` for full training
3. **Evaluate**: Run `scripts/training/evaluate-model.py` to evaluate on test set
4. **Deploy**: Trained model is automatically available on Modal volume

See `scripts/training/README.md` (to be created) for detailed training instructions.

## Notes

- Models are loaded lazily (on first use)
- Models are cached in Modal Volume for faster subsequent loads
- If a model fails to load, the system falls back to heuristic scoring
- All scores are normalized to 0-9 band scale
- Engessay model successfully differentiates between essay quality levels
- Custom trained models (corpus-roberta) are stored on Modal volume and loaded from there
