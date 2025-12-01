"""API middleware."""

import os
import re
from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

# Public paths that don't require authentication
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

# Regex pattern for Authorization header: "Token <key>"
AUTH_TOKEN_PATTERN = re.compile(r"^Token\s+(.+)$")


def get_api_key() -> str | None:
    """Get API key from environment."""
    api_key = os.getenv("MODAL_API_KEY")
    if not api_key:
        print("⚠️ WARNING: MODAL_API_KEY not set. Service will reject all requests.")
        print("   Set it via: modal secret create MODAL_API_KEY <your-key>")
    return api_key


def _unauthorized_response(detail: str) -> JSONResponse:
    """Create an unauthorized response."""
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": detail},
    )


def _server_error_response(detail: str) -> JSONResponse:
    """Create a server error response."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": detail},
    )


async def verify_api_key(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Verify API key from Authorization header."""
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return _unauthorized_response("Missing Authorization header")

    match = AUTH_TOKEN_PATTERN.match(auth_header)
    if not match:
        return _unauthorized_response(
            "Invalid Authorization header format. Expected: 'Token <key>'"
        )

    provided_key = match.group(1)
    expected_key = get_api_key()

    if not expected_key:
        return _server_error_response("Server configuration error: API key not configured")

    if provided_key != expected_key:
        return _unauthorized_response("Invalid API key")

    return await call_next(request)
