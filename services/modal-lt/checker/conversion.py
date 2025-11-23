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
    category = match.category if hasattr(match, "category") else "UNKNOWN"
    rule_id = match.ruleId if hasattr(match, "ruleId") else "UNKNOWN"
    issue_type = determine_issue_type(category, rule_id)

    return {
        "message": match.message,
        "shortMessage": match.message[:50] if len(match.message) > 50 else match.message,
        "offset": match.offset,
        "length": match.errorLength,
        "replacements": [{"value": r} for r in match.replacements[:5]],
        "rule": {
            "id": rule_id,
            "description": match.message,
            "category": {"id": category, "name": category},
        },
        "issueType": issue_type,
    }
