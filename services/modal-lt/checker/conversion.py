"""Match conversion utilities."""

from typing import Any, Final

from .constants import (
    GRAMMAR_CATEGORIES,
    GRAMMAR_RULE_PREFIXES,
    MORFOLOGIK_PREFIX,
    SPELLING_CATEGORIES,
    STYLE_RULE_PREFIXES,
)

# Maximum length for short message
MAX_SHORT_MESSAGE_LENGTH: Final[int] = 50

# Maximum number of replacement suggestions to include
MAX_REPLACEMENTS: Final[int] = 5


def _get_attr_with_fallback(obj: Any, snake_case: str, camel_case: str, default: Any) -> Any:
    """Get attribute with fallback to camelCase if snake_case not found."""
    return getattr(obj, snake_case, getattr(obj, camel_case, default))


def _is_grammar_rule(rule_id: str) -> bool:
    """Check if rule_id indicates a grammar rule."""
    return any(rule_id.startswith(prefix) for prefix in GRAMMAR_RULE_PREFIXES)


def _is_style_rule(rule_id: str) -> bool:
    """Check if rule_id indicates a style rule."""
    return any(rule_id.startswith(prefix) for prefix in STYLE_RULE_PREFIXES)


def _is_error_type(category: str, rule_id: str) -> bool:
    """Check if match indicates an error (vs warning)."""
    return (
        category in GRAMMAR_CATEGORIES
        or _is_grammar_rule(rule_id)
        or category in SPELLING_CATEGORIES
        or rule_id.startswith(MORFOLOGIK_PREFIX)
    )


def determine_issue_type(category: str, rule_id: str) -> str:
    """Determine issue type (error or warning) based on category and rule."""
    if _is_error_type(category, rule_id):
        return "error"
    return "warning"


def convert_match_to_dict(match: Any) -> dict[str, Any]:
    """Convert LanguageTool match to dictionary."""
    # Handle both camelCase and snake_case attribute names for compatibility
    category = getattr(match, "category", "UNKNOWN")
    rule_id = _get_attr_with_fallback(match, "rule_id", "ruleId", "UNKNOWN")
    error_length = _get_attr_with_fallback(match, "error_length", "errorLength", 0)
    offset = getattr(match, "offset", 0)
    message = getattr(match, "message", "")
    replacements = getattr(match, "replacements", [])

    issue_type = determine_issue_type(category, rule_id)
    short_message = (
        message[:MAX_SHORT_MESSAGE_LENGTH] if len(message) > MAX_SHORT_MESSAGE_LENGTH else message
    )

    return {
        "message": message,
        "shortMessage": short_message,
        "offset": offset,
        "length": error_length,
        "replacements": [{"value": r} for r in replacements[:MAX_REPLACEMENTS]],
        "rule": {
            "id": rule_id,
            "description": message,
            "category": {"id": category, "name": category},
        },
        "issueType": issue_type,
    }
