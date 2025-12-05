"""FastAPI application for T-AES-FEEDBACK model."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from model_loader import get_feedback_model


class FeedbackRequest(BaseModel):
    """Request schema for feedback scoring."""

    text: str


class ErrorSpan(BaseModel):
    """Error span detected in text."""

    start: int
    tokens: list[str]


class FeedbackResponse(BaseModel):
    """Response schema for feedback scoring."""

    cefr_score: float
    cefr_level: str
    error_spans: list[ErrorSpan]
    error_types: dict[str, float]


def create_fastapi_app() -> FastAPI:
    """Create and configure FastAPI app."""
    app = FastAPI(
        title="T-AES-FEEDBACK API",
        description="Multi-task CEFR scoring and error detection",
        version="1.0.0",
    )

    # Load model on startup
    model, tokenizer = get_feedback_model()

    @app.post("/score", response_model=FeedbackResponse)
    async def score_essay(request: FeedbackRequest) -> FeedbackResponse:
        """Score essay with T-AES-FEEDBACK model."""
        import torch

        try:
            # Tokenize
            encoding = tokenizer(
                request.text, max_length=512, truncation=True, return_tensors="pt"
            )

            # Move to GPU if available
            device = next(model.parameters()).device
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # Inference
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                )

            # Extract predictions
            cefr_score = outputs["cefr_score"].item()
            span_logits = outputs["span_logits"][0]  # [seq_len, 3]
            error_type_logits = outputs["error_type_logits"][0]  # [5]

            # Convert span predictions to labels
            span_preds = torch.argmax(span_logits, dim=1).tolist()
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

            # Extract error spans
            errors = []
            current_error = None
            for i, (token, label) in enumerate(zip(tokens, span_preds)):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue

                if label == 1:  # B-ERROR
                    if current_error:
                        errors.append(
                            ErrorSpan(
                                start=current_error["start"],
                                tokens=current_error["tokens"],
                            )
                        )
                    current_error = {"start": i, "tokens": [token]}
                elif label == 2 and current_error:  # I-ERROR
                    current_error["tokens"].append(token)
                elif current_error:
                    errors.append(
                        ErrorSpan(
                            start=current_error["start"], tokens=current_error["tokens"]
                        )
                    )
                    current_error = None

            if current_error:
                errors.append(
                    ErrorSpan(
                        start=current_error["start"], tokens=current_error["tokens"]
                    )
                )

            # Error type predictions
            error_type_probs = torch.sigmoid(error_type_logits).tolist()
            error_types = {
                "grammar": error_type_probs[0],
                "vocabulary": error_type_probs[1],
                "mechanics": error_type_probs[2],
                "fluency": error_type_probs[3],
                "other": error_type_probs[4],
            }

            # Convert score to CEFR level
            cefr_level = _score_to_level(cefr_score)

            return FeedbackResponse(
                cefr_score=round(cefr_score, 2),
                cefr_level=cefr_level,
                error_spans=errors,
                error_types=error_types,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "model": "T-AES-FEEDBACK", "epoch": 3}

    return app


def _score_to_level(score: float) -> str:
    """Convert numeric score to CEFR level."""
    if score < 1.5:
        return "A1"
    elif score < 2.5:
        return "A2"
    elif score < 3.5:
        return "A2+"
    elif score < 4.5:
        return "B1"
    elif score < 5.5:
        return "B1+"
    elif score < 6.5:
        return "B2"
    elif score < 7.5:
        return "B2+"
    elif score < 8.5:
        return "C1"
    else:
        return "C2"
