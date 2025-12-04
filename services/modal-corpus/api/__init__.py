"""API initialization and FastAPI app factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router


def create_fastapi_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Writeo Corpus CEFR Scorer",
        description="CEFR scoring using corpus-trained RoBERTa model",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    return app
