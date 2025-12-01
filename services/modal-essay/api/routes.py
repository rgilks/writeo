"""API route handlers."""


def handle_health() -> dict[str, str]:
    """Handle health check endpoint."""
    return {"status": "ok"}
