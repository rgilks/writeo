"""API route handlers and endpoints."""

import torch  # type: ignore[import-untyped]
from fastapi import APIRouter, HTTPException

from config import MODEL_NAME, score_to_cefr
from model_loader import get_model
from schemas import ModelInfo, ScoreRequest, ScoreResponse

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_NAME}


@router.get("/model/info")
def model_info() -> ModelInfo:
    """Get model information."""
    return ModelInfo(
        name="corpus-roberta",
        description="RoBERTa-base trained on Write & Improve corpus for CEFR scoring",
        version="1.0.0",
    )


@router.post("/score")
def score_essay(request: ScoreRequest) -> ScoreResponse:
    """Score an essay and return CEFR level."""
    try:
        # Get model and tokenizer
        model, tokenizer = get_model()

        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=request.max_length,
            padding=True,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.squeeze().item()

        # Clip to valid range
        score = max(2.0, min(8.5, score))

        # Convert to CEFR level
        cefr_level = score_to_cefr(score)

        return ScoreResponse(score=round(score, 2), cefr_level=cefr_level, model=MODEL_NAME)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}") from e
