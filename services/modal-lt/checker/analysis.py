"""Match analysis utilities."""

from collections import Counter
from typing import Any

from .constants import (
    GRAMMAR_CATEGORIES,
    GRAMMAR_RULE_PREFIXES,
    MORFOLOGIK_PREFIX,
    SPELLING_CATEGORIES,
    STYLE_CATEGORIES,
)


def _get_match_attributes(match: Any) -> tuple[str, str]:
    """Extract category and rule_id from match, handling both naming conventions."""
    category = getattr(match, "category", "UNKNOWN")
    rule_id = getattr(match, "rule_id", getattr(match, "ruleId", "UNKNOWN"))
    return category, rule_id


def _is_grammar_rule(rule_id: str) -> bool:
    """Check if rule_id indicates a grammar rule."""
    return any(rule_id.startswith(prefix) for prefix in GRAMMAR_RULE_PREFIXES)


def _is_spelling_rule(category: str, rule_id: str) -> bool:
    """Check if match indicates a spelling error."""
    return category in SPELLING_CATEGORIES or rule_id.startswith(MORFOLOGIK_PREFIX)


def _categorize_match(category: str, rule_id: str) -> str:
    """Categorize a match as grammar, spelling, style, or unknown."""
    if category in GRAMMAR_CATEGORIES or _is_grammar_rule(rule_id):
        return "grammar"
    if _is_spelling_rule(category, rule_id):
        return "spelling"
    if category in STYLE_CATEGORIES:
        return "style"
    return "unknown"


def analyze_matches(matches: list[Any]) -> dict[str, Any]:
    """Analyze match types and return statistics."""
    categories: Counter[str] = Counter()
    rule_types: Counter[str] = Counter()
    counts = {"grammar": 0, "spelling": 0, "style": 0, "unknown": 0}

    for match in matches:
        category, rule_id = _get_match_attributes(match)
        categories[category] += 1
        rule_types[rule_id] += 1
        match_type = _categorize_match(category, rule_id)
        counts[match_type] += 1

    print(f"📊 Categories found: {dict(categories)}")
    print(
        f"📊 Breakdown: Grammar={counts['grammar']}, "
        f"Spelling={counts['spelling']}, "
        f"Style={counts['style']}, "
        f"Unknown={counts['unknown']}"
    )
    print(f"📋 Unique rule IDs: {len(rule_types)}")
    rule_ids_list = list(rule_types.keys())
    if len(rule_ids_list) <= 15:
        print(f"📋 Rule IDs: {rule_ids_list}")
    else:
        print(f"📋 First 15 Rule IDs: {rule_ids_list[:15]}")

    if counts["grammar"] == 0 and counts["spelling"] > 0:
        print("⚠️  WARNING: Only spelling errors detected, no grammar errors found!")

    return {
        "categories": dict(categories),
        "rule_types": dict(rule_types),
        "grammar_count": counts["grammar"],
        "spelling_count": counts["spelling"],
        "style_count": counts["style"],
        "unknown_count": counts["unknown"],
    }
