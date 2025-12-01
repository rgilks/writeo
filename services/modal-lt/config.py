"""LanguageTool configuration constants."""

from typing import Final

# LanguageTool version (used in API responses)
LT_VERSION: Final[str] = "6.4"

# Directory paths for LanguageTool data storage
# Note: These paths must match VOLUME_MOUNT_* constants in app.py
LT_CACHE_DIR: Final[str] = "/vol/lt-cache"  # LanguageTool cache directory
LT_JAR_DIR: Final[str] = "/vol/lt-jar"  # LanguageTool JAR file directory
LT_NGRAM_DIR: Final[str] = "/vol/lt-ngrams"  # N-gram data directory

# Base URL for downloading LanguageTool n-gram data
NGRAM_BASE_URL: Final[str] = "https://languagetool.org/download/ngram-data/"
