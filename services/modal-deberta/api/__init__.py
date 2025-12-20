"""API module for DeBERTa AES service."""

import time
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import MODEL_DISPLAY_NAME, score_to_cefr
from model_loader import get_model
from schemas import HealthResponse, ScoreRequest


def create_fastapi_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="AES-DEBERTA API",
        description="DeBERTa-v3-large essay scoring with dimensional outputs",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model=MODEL_DISPLAY_NAME,
            gpu_available=torch.cuda.is_available(),
        )

    @app.post("/score", response_model=dict[str, Any])
    async def score_essay(request: ScoreRequest):
        """
        Score an essay and return dimensional + overall scores.

        Returns:
            {
                "type": "grader",
                "overall": 7.0,
                "label": "C1",
                "dimensions": {
                    "TA": 7.5,
                    "CC": 7.0,
                    "Vocab": 7.0,
                    "Grammar": 7.0,
                    "Overall": 7.0
                }
            }
        """
        try:
            start_time = time.time()

            # Get model
            model, tokenizer = get_model()

            # Tokenize
            encoded = tokenizer(
                request.text,
                max_length=request.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            device = next(model.parameters()).device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Inference
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)

            # Extract scores
            ta = float(outputs["ta"].cpu().item())
            cc = float(outputs["cc"].cpu().item())
            vocab = float(outputs["vocab"].cpu().item())
            grammar = float(outputs["grammar"].cpu().item())
            overall = float(outputs["overall"].cpu().item())
            cefr_score = float(outputs["cefr_score"].cpu().item())

            # Clamp scores to valid range
            ta = max(0.0, min(9.0, ta))
            cc = max(0.0, min(9.0, cc))
            vocab = max(0.0, min(9.0, vocab))
            grammar = max(0.0, min(9.0, grammar))
            overall = max(0.0, min(9.0, overall))
            cefr_score = max(2.0, min(9.0, cefr_score))

            # Round to 0.5 increments
            def round_to_half(x: float) -> float:
                return round(x * 2) / 2

            ta = round_to_half(ta)
            cc = round_to_half(cc)
            vocab = round_to_half(vocab)
            grammar = round_to_half(grammar)
            overall = round_to_half(overall)

            # Get CEFR level
            cefr_level = score_to_cefr(cefr_score)

            inference_time = time.time() - start_time

            return {
                "type": "grader",
                "overall": overall,
                "label": cefr_level,
                "dimensions": {
                    "TA": ta,
                    "CC": cc,
                    "Vocab": vocab,
                    "Grammar": grammar,
                    "Overall": overall,
                },
                "metadata": {
                    "model": MODEL_DISPLAY_NAME,
                    "inference_time_ms": round(inference_time * 1000, 2),
                },
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app
