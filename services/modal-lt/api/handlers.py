"""API route handlers."""

import sys
import time

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from checker import check_text_with_languagetool
from config import LT_VERSION
from schemas import CheckRequest
from tool_loader import get_languagetool_tool, lt_tool


async def handle_health() -> dict:
    """Handle health check endpoint."""
    try:
        get_languagetool_tool("en-GB")
        return {
            "status": "ok",
            "lt_version": LT_VERSION,
            "tool_initialized": lt_tool is not None,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def handle_check(request: CheckRequest) -> JSONResponse:
    """Handle grammar check endpoint."""
    request_start = time.time()
    print(f"\n{'=' * 60}")
    print("📥 POST /check request received")
    print(f"⏱️  Request time: {request_start:.2f}s")
    print(f"🌐 Language: {request.language}")
    print(f"📝 Text length: {len(request.text) if request.text else 0} chars")
    if request.answer_id:
        print(f"🆔 Answer ID: {request.answer_id}")

    try:
        if not request.text or not request.text.strip():
            print("⚠️  Empty text provided, returning empty matches")
            return JSONResponse(
                content={
                    "software": {"name": "LanguageTool", "version": LT_VERSION},
                    "language": {"name": "English (GB)", "code": request.language},
                    "matches": [],
                    "meta": {
                        "textLength": 0,
                        "wordCount": 0,
                        "checkTimeMs": 0,
                        "totalMatches": 0,
                    },
                },
                status_code=200,
            )

        result = check_text_with_languagetool(request.text, request.language)

        request_time = time.time() - request_start
        print(f"✅ Request completed in {request_time:.3f}s")
        print(
            f"📊 Returning {result.get('meta', {}).get('totalMatches', len(result.get('matches', [])))} matches"
        )
        print(f"{'=' * 60}\n")

        return JSONResponse(content=result, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        request_time = time.time() - request_start
        error_msg = f"Internal server error after {request_time:.3f}s: {str(e)}"
        print(f"❌ {error_msg}", file=sys.stderr)
        import traceback

        print(f"📜 Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=error_msg) from None
