"""API middleware."""

import os
import re
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import status


def get_api_key() -> str:
    """Get API key from environment."""
    MODAL_API_KEY = os.getenv("MODAL_API_KEY")
    if not MODAL_API_KEY:
        print("⚠️ WARNING: MODAL_API_KEY not set. Service will reject all requests.")
        print("   Set it via: modal secret create MODAL_API_KEY <your-key>")
    return MODAL_API_KEY


async def verify_api_key(request: Request, call_next):
    """Verify API key from Authorization header."""
    path = request.url.path
    if path in ["/health", "/docs", "/openapi.json", "/redoc"]:
        return await call_next(request)
    
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Missing Authorization header"}
        )
    
    match = re.match(r"^Token\s+(.+)$", auth_header)
    if not match:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid Authorization header format. Expected: 'Token <key>'"}
        )
    
    provided_key = match.group(1)
    MODAL_API_KEY = get_api_key()
    
    if not MODAL_API_KEY:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Server configuration error: API key not configured"}
        )
    
    if provided_key != MODAL_API_KEY:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid API key"}
        )
    
    return await call_next(request)

