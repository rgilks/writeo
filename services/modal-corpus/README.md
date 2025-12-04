# Modal Corpus CEFR Scorer

REST API service for CEFR scoring using a corpus-trained RoBERTa model.

## Service Information

**Deployed URL**: https://rob-gilks--writeo-corpus-fastapi-app.modal.run

## Endpoints

### Health Check

```bash
GET /health
```

**Response**: `{"status": "ok", "model": "corpus-roberta"}`

### Model Information

```bash
GET /model/info
```

**Response**:

```json
{
  "name": "corpus-roberta",
  "description": "RoBERTa-base trained on Write & Improve corpus for CEFR scoring",
  "version": "1.0.0"
}
```

### Score Essay

```bash
POST /score
Content-Type: application/json

{
  "text": "Essay text to score",
  "max_length": 512
}
```

**Response**:

```json
{
  "score": 3.74,
  "cefr_level": "A2+",
  "model": "corpus-roberta"
}
```

## Model Details

**Architecture**: RoBERTa-base (125M parameters)  
**Training Data**: Write & Improve corpus (4,741 essays)  
**Training Results**: Train loss 0.27, Eval loss 0.43  
**Model Size**: 498MB  
**CEFR Range**: A1 (2.0) to C2 (8.5)  
**Storage**: Modal volume `/vol/models/corpus-trained-roberta`

## Deployment

### Deploy to Modal

```bash
cd services/modal-corpus
modal deploy app.py
```

### Local Development

```bash
cd services/modal-corpus
modal serve app.py
```

## Configuration

Edit `config.py` to modify:

- Model path on volume
- CEFR score mapping
- Score to CEFR level conversion

## Files

- `app.py` - Modal app configuration
- `config.py` - CEFR mapping and paths
- `model_loader.py` - Model loading from volume
- `schemas.py` - Pydantic request/response models
- `api/routes.py` - FastAPI endpoint handlers
- `api/__init__.py` - App factory

## Infrastructure

**GPU**: T4  
**Memory**: 4GB  
**Timeout**: 60 seconds  
**Scaledown**: 30 seconds  
**Volume**: `writeo-models` (shared with training)

## Training

To retrain this model, see `scripts/training/README.md`.
