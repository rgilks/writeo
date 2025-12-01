"""Match conversion utilities."""

from typing import Any

from .constants import (
    GRAMMAR_CATEGORIES,
    GRAMMAR_RULE_PREFIXES,
    SPELLING_CATEGORIES,
    STYLE_CATEGORIES,
    STYLE_RULE_PREFIXES,
)


def determine_issue_type(category: str, rule_id: str) -> str:
    """Determine issue type (error or warning) based on category and rule."""
    is_grammar_rule = any(rule_id.startswith(prefix) for prefix in GRAMMAR_RULE_PREFIXES)
    is_style_rule = any(rule_id.startswith(prefix) for prefix in STYLE_RULE_PREFIXES)

    if (
        category in GRAMMAR_CATEGORIES
        or is_grammar_rule
        or category in SPELLING_CATEGORIES
        or rule_id.startswith("MORFOLOGIK")
    ):
        return "error"
    elif category in STYLE_CATEGORIES or is_style_rule:
        return "warning"
    else:
        return "warning"


def convert_match_to_dict(match: Any) -> dict[str, Any]:
    """Convert LanguageTool match to dictionary."""
    # Handle both camelCase and snake_case attribute names for compatibility
    category = getattr(match, "category", "UNKNOWN")
    rule_id = getattr(match, "rule_id", getattr(match, "ruleId", "UNKNOWN"))
    error_length = getattr(match, "error_length", getattr(match, "errorLength", 0))
    offset = getattr(match, "offset", 0)
    message = getattr(match, "message", "")
    replacements = getattr(match, "replacements", [])

    issue_type = determine_issue_type(category, rule_id)

    return {
        "message": message,
        "shortMessage": message[:50] if len(message) > 50 else message,
        "offset": offset,
        "length": error_length,
        "replacements": [{"value": r} for r in replacements[:5]],
        "rule": {
            "id": rule_id,
            "description": message,
            "category": {"id": category, "name": category},
        },
        "issueType": issue_type,
    }
