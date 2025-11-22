"""Model management handlers."""

from typing import Dict, Any
from schemas import ModalRequest
from config import DEFAULT_MODEL, MODEL_CONFIGS
from model_loader import get_model
from scoring import score_essay


async def handle_list_models() -> Dict[str, Any]:
    """Handle list models endpoint."""
    models_status = {}
    for key, config in MODEL_CONFIGS.items():
        status = "unknown"
        try:
            if key == "fallback":
                status = "available"
            else:
                model, tokenizer = get_model(key)
                if model is not None and tokenizer is not None:
                    status = "loaded"
                else:
                    status = "error"
        except Exception as e:
            status = f"error: {str(e)[:50]}"
        
        models_status[key] = {
            "name": config["name"],
            "type": config["type"],
            "status": status,
            "is_default": key == DEFAULT_MODEL
        }
    
    return {"models": models_status, "default": DEFAULT_MODEL}


async def handle_compare_models(request: ModalRequest) -> Dict[str, Any]:
    """Handle compare models endpoint."""
    results: Dict[str, Any] = {}
    
    for model_key in MODEL_CONFIGS.keys():
        if model_key == "fallback":
            continue
        
        try:
            model, tokenizer = get_model(model_key)
            if model is None or tokenizer is None:
                continue
            
            if request.parts and request.parts[0].answers:
                answer = request.parts[0].answers[0]
                scores = score_essay(
                    answer.question_text,
                    answer.answer_text,
                    model,
                    tokenizer,
                    model_key=model_key
                )
                results[model_key] = scores
        except Exception as e:
            results[model_key] = {"error": str(e)}
    
    return {"comparison": results}

