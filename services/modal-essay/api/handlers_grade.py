"""Grade endpoint handler."""

import time
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
        if model_key is None:
            model_key = DEFAULT_MODEL

        response = process_submission(request, model_key)
        result_dict = response.model_dump(mode="json", exclude_none=True, by_alias=True)

        request_time = time.time() - request_start
        print(f"✅ Request completed in {request_time:.2f}s")
        print(f"{'=' * 60}\n")

        return result_dict  # type: ignore[no-any-return]
    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"❌ ERROR in /grade endpoint: {e}")
        print("❌ Full traceback:")
        print(error_trace)

        error_message = f"{type(e).__name__}: {str(e)}"
        error_response = AssessmentResults(
            status="error",
            template=request.template
            if "request" in locals()
            else {"name": "unknown", "version": 1},
            error_message=error_message,
        )
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode="json", exclude_none=True, by_alias=True),
        )
