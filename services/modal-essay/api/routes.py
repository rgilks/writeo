"""API route handlers - main exports."""

from typing import Dict
from .handlers_grade import handle_grade
from .handlers_models import handle_list_models, handle_compare_models


async def handle_health() -> Dict[str, str]:
    """Handle health check endpoint."""
    return {"status": "ok"}

