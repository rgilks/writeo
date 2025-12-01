"""Grade endpoint handler."""

import time
import traceback
from typing import Any

from fastapi.responses import JSONResponse

from config import DEFAULT_MODEL
from schemas import AssessmentResults, ModalRequest

from .handlers_submission import process_submission


async def handle_grade(
    request: ModalRequest, model_key: str | None = None
) -> dict[str, Any] | JSONResponse:
    """Handle grade endpoint request."""
    request_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"📥 POST /grade request received (t={request_start:.2f}s)")

    try:
        model_key = model_key or DEFAULT_MODEL
        response = process_submission(request, model_key)
        result: dict[str, Any] = response.model_dump(mode="json", exclude_none=True, by_alias=True)

        request_time = time.time() - request_start
        print(f"✅ Request completed in {request_time:.2f}s")
        print(f"{'=' * 60}\n")

        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ ERROR in /grade endpoint: {e}")
        print("❌ Full traceback:")
        print(error_trace)

        error_response = AssessmentResults(
            status="error",
            template=request.template,
            error_message=f"{type(e).__name__}: {str(e)}",
        )
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json", exclude_none=True, by_alias=True),
        )
