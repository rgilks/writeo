"""Constants for text checking."""

from typing import Final

# Maximum text length for performance (characters)
MAX_TEXT_LENGTH: Final[int] = 10000

# Rule ID prefix for Morfologik spelling checker
MORFOLOGIK_PREFIX: Final[str] = "MORFOLOGIK"

# Grammar rule ID prefixes (LanguageTool rule IDs that start with these indicate grammar issues)
GRAMMAR_RULE_PREFIXES: Final[tuple[str, ...]] = (
    "SVA",
    "AGREEMENT",
    "CONFUSION",
    "CONFUSED_WORDS",
    "TENSE",
    "VERB",
    "PREPOSITION",
    "ARTICLE",
    "PRONOUN",
    "PLURAL",
    "POSSESSIVE",
    "COMPARISON",
    "CONJUNCTION",
    "MODAL",
    "PASSIVE",
    "GERUND",
    "INFINITIVE",
    "PARTICIPLE",
    "CLAUSE",
    "FRAGMENT",
    "RUN_ON",
)

# LanguageTool categories that indicate grammar issues
GRAMMAR_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "GRAMMAR",
        "TYPOS",
        "CONFUSED_WORDS",
        "TYPOGRAPHY",
        "CASING",
    }
)

# LanguageTool categories that indicate style issues
STYLE_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "STYLE",
        "REDUNDANCY",
        "PUNCTUATION",
        "SEMANTICS",
    }
)

# LanguageTool categories that indicate spelling issues
SPELLING_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "MORFOLOGIK",
        "SPELLING",
        "MORFOLOGIK_RULE",
    }
)

# Style rule ID prefixes (LanguageTool rule IDs that start with these indicate style issues)
STYLE_RULE_PREFIXES: Final[tuple[str, ...]] = (
    "REDUNDANCY",
    "WORDINESS",
    "REPETITION",
    "CLARITY",
    "FORMALITY",
)
