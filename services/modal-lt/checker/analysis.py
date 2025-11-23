"""Match analysis utilities."""

from typing import Any

from .constants import (
    GRAMMAR_CATEGORIES,
    GRAMMAR_RULE_PREFIXES,
    SPELLING_CATEGORIES,
    STYLE_CATEGORIES,
)


def analyze_matches(matches: list) -> dict[str, Any]:
    """Analyze match types and return statistics."""
    categories = {}
    rule_types = {}
    grammar_count = 0
    spelling_count = 0
    style_count = 0
    unknown_count = 0

    for match in matches:
        cat = match.category if hasattr(match, "category") else "UNKNOWN"
        categories[cat] = categories.get(cat, 0) + 1
        rule_id = match.ruleId if hasattr(match, "ruleId") else "UNKNOWN"
        rule_types[rule_id] = rule_types.get(rule_id, 0) + 1

        is_grammar_rule = any(rule_id.startswith(p) for p in GRAMMAR_RULE_PREFIXES)
        if cat in GRAMMAR_CATEGORIES or is_grammar_rule:
            grammar_count += 1
        elif cat in SPELLING_CATEGORIES or rule_id.startswith("MORFOLOGIK"):
            spelling_count += 1
        elif cat in STYLE_CATEGORIES:
            style_count += 1
        else:
            unknown_count += 1

    print(f"📊 Categories found: {dict(categories)}")
    print(
        f"📊 Breakdown: Grammar={grammar_count}, Spelling={spelling_count}, Style={style_count}, Unknown={unknown_count}"
    )
    print(f"📋 Unique rule IDs: {len(rule_types)}")
    if len(rule_types) <= 15:
        print(f"📋 Rule IDs: {list(rule_types.keys())}")
    else:
        print(f"📋 First 15 Rule IDs: {list(rule_types.keys())[:15]}")

    if grammar_count == 0 and spelling_count > 0:
        print("⚠️  WARNING: Only spelling errors detected, no grammar errors found!")

    return {
        "categories": categories,
        "rule_types": rule_types,
        "grammar_count": grammar_count,
        "spelling_count": spelling_count,
        "style_count": style_count,
        "unknown_count": unknown_count,
    }
