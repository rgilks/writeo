"""API module - FastAPI app creation."""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from schemas import ModalRequest, AssessmentResults
from config import DEFAULT_MODEL
from model_loader import get_model
from .middleware import verify_api_key
from .routes import handle_health
from .handlers_grade import handle_grade as handle_grade_impl
from .handlers_models import handle_list_models as handle_list_models_impl, handle_compare_models as handle_compare_models_impl


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    startup_start = time.time()
    try:
        print("=" * 60)
        print("🚀 Pre-loading default model at startup...")
        print(f"⏱️  Startup began at: {startup_start:.2f}s")
        model_load_start = time.time()
        _ = get_model(DEFAULT_MODEL)
        model_load_time = time.time() - model_load_start
        total_startup_time = time.time() - startup_start
        print(f"✅ Default model ({DEFAULT_MODEL}) pre-loaded successfully!")
        print(f"⏱️  Model load time: {model_load_time:.2f}s")
        print(f"⏱️  Total startup time: {total_startup_time:.2f}s")
        print("=" * 60)
    except Exception as e:
        total_startup_time = time.time() - startup_start
        print(f"⚠️ Warning: Could not pre-load default model after {total_startup_time:.2f}s: {e}")
        print("Models will be loaded on first request (lazy loading)")
        import traceback
        print(f"📜 Traceback:\n{traceback.format_exc()}")

    yield

    print("🛑 Shutting down essay scoring service...")


def create_fastapi_app() -> FastAPI:
    """Create and configure FastAPI app."""
    api = FastAPI(
        title="Writeo Essay Scorer API",
        description="""
        ## Writeo Essay Scoring Service
        
        A high-performance API for automated essay scoring using transformer-based machine learning models.
        
        ### Features
        - **Multiple Model Support**: Choose from different scoring models (engessay, distilbert, fallback)
        - **Band Scoring**: Returns scores on the 0-9 band scale with 0.5 increments
        - **CEFR Mapping**: Automatic conversion from band scores to CEFR levels (A2-C2)
        - **Multi-Dimensional Scoring**: Provides scores for Task Achievement (TA), Coherence & Cohesion (CC), Vocabulary, and Grammar
        - **GPU Acceleration**: Fast inference using GPU-accelerated transformer models
        
        ### Models
        - **engessay** (default): KevSun/Engessay_grading_ML - RoBERTa-based model with 6 analytic dimensions
          - Citation: Sun, K., & Wang, R. (2024). Automatic Essay Multi-dimensional Scoring with Fine-tuning and Multiple Regression. *ArXiv*. https://arxiv.org/abs/2406.01198
        - **distilbert**: Michau96/distilbert-base-uncased-essay_scoring - DistilBERT model with single score output
        - **fallback**: Heuristic-based scoring (no ML model required)
        
        ### Scoring Dimensions
        - **TA** (Task Achievement): How well the task is addressed
        - **CC** (Coherence & Cohesion): Organization and linking of ideas
        - **Vocab** (Vocabulary): Range and accuracy of vocabulary
        - **Grammar**: Grammatical range and accuracy
        - **Overall**: Average of all dimensions, mapped to CEFR level
        
        ### Access
        - **Swagger UI**: Available at `/docs`
        - **ReDoc**: Available at `/redoc`
        - **OpenAPI JSON**: Available at `/openapi.json`
        """,
        version="1.0.0",
        contact={
            "name": "Robert Gilks",
            "url": "https://tre.systems",
        },
        license_info={
            "name": "Apache-2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        tags_metadata=[
            {
                "name": "Assessment",
                "description": "Essay scoring and assessment endpoints",
            },
            {
                "name": "Models",
                "description": "Model management and comparison",
            },
            {
                "name": "Health",
                "description": "Health check and system status",
            },
        ],
        lifespan=lifespan,
    )
    
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    api.middleware("http")(verify_api_key)
    
    register_routes(api)
    
    return api


def register_routes(api: FastAPI) -> None:
    """Register API routes."""
    from typing import Optional
    from fastapi import Query
    
    @api.post(
        "/grade",
        response_model=AssessmentResults,
        status_code=200,
        tags=["Assessment"],
        summary="Grade essay submission",
        description="Scores an essay submission using machine learning models."
    )
    async def grade(
        request: ModalRequest,
        model_key: Optional[str] = Query(
            None,
            description="Model to use: 'engessay' (default), 'distilbert', or 'fallback'",
            example="engessay"
        )
    ):
        return await handle_grade_impl(request, model_key)
    
    @api.get("/health", tags=["Health"], summary="Health check")
    async def health():
        return await handle_health()
    
    @api.get("/models", tags=["Models"], summary="List available models")
    async def list_models():
        return await handle_list_models_impl()
    
    @api.post("/grade/compare", tags=["Models"], summary="Compare models on same submission")
    async def compare_models(request: ModalRequest):
        return await handle_compare_models_impl(request)

