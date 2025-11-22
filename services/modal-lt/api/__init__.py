"""API module - FastAPI app creation."""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schemas import CheckRequest
from tool_loader import get_languagetool_tool
from .middleware import verify_api_key, get_api_key
from .handlers import handle_health, handle_check


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    startup_start = time.time()
    try:
        print("=" * 60)
        print("🚀 Pre-initializing LanguageTool on container startup...")
        print(f"⏱️  Startup began at: {startup_start:.2f}s")
        get_languagetool_tool("en-GB")
        startup_time = time.time() - startup_start
        print(f"✅ LanguageTool pre-initialized successfully in {startup_time:.2f}s")
        print("=" * 60)
    except Exception as e:
        startup_time = time.time() - startup_start
        print(f"⚠️  Warning: Failed to pre-initialize LanguageTool after {startup_time:.2f}s: {e}")
        import traceback

        print(f"📜 Traceback:\n{traceback.format_exc()}")

    yield

    print("🛑 Shutting down LanguageTool service...")


def create_fastapi_app() -> FastAPI:
    """Create and configure FastAPI app."""
    api = FastAPI(
        title="Writeo LanguageTool Service",
        description="Grammar checking service using LanguageTool",
        version="0.2.1",
        lifespan=lifespan,
    )

    get_api_key()
    api.middleware("http")(verify_api_key)

    @api.get("/health", tags=["Health"])
    async def health():
        """Health check endpoint."""
        return await handle_health()

    @api.post("/check", tags=["Grammar Check"])
    async def check(request: CheckRequest):
        """Check text for grammar and language errors using LanguageTool."""
        return await handle_check(request)

    return api

