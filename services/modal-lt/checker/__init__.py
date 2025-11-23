"""Text checking module."""

import sys
import time
from typing import Any

from fastapi import HTTPException

from config import LT_VERSION
from tool_loader import get_languagetool_tool

from .analysis import analyze_matches
from .constants import MAX_TEXT_LENGTH
from .conversion import convert_match_to_dict


def truncate_text(text: str) -> str:
    """Truncate text if too long."""
    original_length = len(text)
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        print(
            f"⚠️  Text truncated from {original_length} to {MAX_TEXT_LENGTH} characters for performance"
        )
    return text


def create_result_base(
    language: str, text: str, check_time_elapsed: float, matches: list, categories: dict[str, int]
) -> dict[str, Any]:
    """Create base result structure."""
    return {
        "software": {"name": "LanguageTool", "version": LT_VERSION},
        "language": {
            "name": "English (GB)" if language == "en-GB" else language,
            "code": language,
        },
        "matches": [],
        "meta": {
            "textLength": len(text),
            "wordCount": len(text.split()),
            "checkTimeMs": round(check_time_elapsed * 1000, 2),
            "totalMatches": len(matches),
            "categories": categories,
        },
    }


def check_text_with_languagetool(text: str, language: str = "en-GB") -> dict[str, Any]:
    """Check text using LanguageTool Python library."""
    check_start_time = time.time()
    tool = get_languagetool_tool(language)

    text = truncate_text(text)
    print(f"🔍 Checking text: {len(text)} characters, {len(text.split())} words")

    try:
        check_time_start = time.time()
        matches = tool.check(text)
        check_time_elapsed = time.time() - check_time_start

        print(f"✅ Check completed in {check_time_elapsed:.3f}s: {len(matches)} issues found")

        stats = analyze_matches(matches)
        result = create_result_base(
            language, text, check_time_elapsed, matches, stats["categories"]
        )

        for match in matches:
            result["matches"].append(convert_match_to_dict(match))

        total_time = time.time() - check_start_time
        print(f"⏱️  Total processing time: {total_time:.3f}s")
        result["meta"]["totalTimeMs"] = round(total_time * 1000, 2)

        return result
    except Exception as e:
        error_msg = f"LanguageTool check error: {str(e)}"
        print(f"❌ {error_msg}", file=sys.stderr)
        import traceback

        print(f"📜 Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=error_msg) from None
