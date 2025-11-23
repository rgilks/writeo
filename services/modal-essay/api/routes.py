"""API route handlers - main exports."""


async def handle_health() -> dict[str, str]:
    """Handle health check endpoint."""
    return {"status": "ok"}
