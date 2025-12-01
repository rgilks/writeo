"""LanguageTool tool loading utilities."""

import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from config import LT_CACHE_DIR, LT_JAR_DIR
from ngram_setup import setup_ngram_data

# Constants
BYTES_PER_MB = 1024 * 1024
DIRECTORY_MODE = 0o755
MAX_SPELLING_SUGGESTIONS = 5
MAX_TEST_MATCHES_TO_SHOW = 3
MAX_MESSAGE_PREVIEW_LENGTH = 50
MAX_JAR_FILES_TO_SHOW = 5

# Test texts for validation
TEST_GRAMMAR_TEXT = "I goes to the store. He don't like it."
VALIDATION_TEST_TEXT = "I goes to store"

# Cache directory paths
ROOT_CACHE_DIR = "/root/.cache/language-tool-python"
CACHE_SUBDIR = ".cache/language-tool-python"

# N-gram environment variables
NGRAM_ENV_VARS = ["LANGUAGETOOL_LANGUAGE_MODEL", "LT_LANGUAGE_MODEL"]

# Global variable to store the LanguageTool tool instance
lt_tool: Any | None = None


def setup_environment(cache_base: str, jar_dir: str) -> tuple[str, str | None, str | None]:
    """Set up environment variables for LanguageTool."""
    original_home = os.environ.get("HOME", "/root")
    original_xdg_cache = os.environ.get("XDG_CACHE_HOME")
    original_ltp_path = os.environ.get("LTP_PATH")

    os.environ["LTP_PATH"] = jar_dir
    os.environ["HOME"] = cache_base
    os.environ["XDG_CACHE_HOME"] = cache_base

    print(f"📁 LTP_PATH set to: {jar_dir}")
    print(f"📁 HOME set to: {cache_base}")
    print(f"📁 XDG_CACHE_HOME set to: {cache_base}")

    return original_home, original_xdg_cache, original_ltp_path


def find_jar_in_cache(jar_dir: str, cache_base: str) -> tuple[bool, Path | None]:
    """Find JAR file in cache directories."""
    if Path(jar_dir).exists():
        jar_files = list(Path(jar_dir).rglob("*.jar"))
        if jar_files:
            return True, jar_files[0]

    cache_path = Path(cache_base) / ".cache" / "language-tool-python"
    if cache_path.exists():
        jar_files = list(cache_path.rglob("*.jar"))
        if jar_files:
            return True, jar_files[0]

    return False, None


def _bytes_to_mb(bytes_value: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_value / BYTES_PER_MB


def configure_ngrams(ngram_path: str | None) -> None:
    """Configure n-gram environment variables."""
    if ngram_path:
        for env_var in NGRAM_ENV_VARS:
            os.environ[env_var] = ngram_path
            print(f"✅ Set {env_var}={ngram_path}")
    else:
        for env_var in NGRAM_ENV_VARS:
            if env_var in os.environ:
                del os.environ[env_var]
        print("ℹ️  N-gram data not available - continuing without n-grams")


def _get_match_attribute(
    match: Any, snake_case: str, camel_case: str, default: str = "UNKNOWN"
) -> str:
    """Get attribute from match, handling both naming conventions."""
    return getattr(match, snake_case, getattr(match, camel_case, default))


def create_tool_with_config(language: str, ngram_path: str | None) -> Any:
    """Create LanguageTool instance with configuration."""
    import language_tool_python

    print(f"🧪 Testing grammar detection with: '{TEST_GRAMMAR_TEXT}'")

    tool_config: dict[str, Any] = {"maxSpellingSuggestions": MAX_SPELLING_SUGGESTIONS}

    if ngram_path:
        tool_config["languageModel"] = ngram_path
        print(f"✅ Configuring LanguageTool with n-grams: {ngram_path}")
        print("   Note: n-gram path should point to parent directory (contains 'en/' folder)")

    test_tool = language_tool_python.LanguageTool(language, config=tool_config)
    test_matches = test_tool.check(TEST_GRAMMAR_TEXT)
    print(f"🧪 Test grammar check found {len(test_matches)} issues")

    if test_matches:
        for match in test_matches[:MAX_TEST_MATCHES_TO_SHOW]:
            rule_id = _get_match_attribute(match, "rule_id", "ruleId")
            category = _get_match_attribute(match, "category", "category")
            message = _get_match_attribute(match, "message", "message", "")
            preview = (
                message[:MAX_MESSAGE_PREVIEW_LENGTH]
                if len(message) > MAX_MESSAGE_PREVIEW_LENGTH
                else message
            )
            print(f"   - Rule: {rule_id}, Category: {category}, Message: {preview}")
    else:
        print("⚠️  WARNING: No grammar errors detected in test text!")

    return test_tool


def copy_jar_to_volume(all_jar_locations: list[Path], jar_dir: str) -> None:
    """Copy JAR files to volume if not already there."""
    for jar_file in all_jar_locations:
        if str(jar_file).startswith("/vol"):
            continue

        dest = Path(jar_dir) / jar_file.name
        if not dest.exists():
            print(f"📋 Copying JAR to volume: {dest}")
            shutil.copy2(jar_file, dest)


def restore_environment(
    original_home: str, original_xdg_cache: str | None, original_ltp_path: str | None
) -> None:
    """Restore original environment variables."""
    if original_home:
        os.environ["HOME"] = original_home
    if original_xdg_cache:
        os.environ["XDG_CACHE_HOME"] = original_xdg_cache
    elif "XDG_CACHE_HOME" in os.environ:
        del os.environ["XDG_CACHE_HOME"]
    if original_ltp_path:
        os.environ["LTP_PATH"] = original_ltp_path
    elif "LTP_PATH" in os.environ:
        del os.environ["LTP_PATH"]


def _find_all_jar_locations(jar_dir: str, cache_base: str) -> list[Path]:
    """Find all JAR file locations in cache directories."""
    all_jar_locations = []
    check_dirs: list[str | Path] = [
        jar_dir,
        Path(cache_base) / CACHE_SUBDIR,
        Path(ROOT_CACHE_DIR),
    ]

    for check_dir in check_dirs:
        check_path = Path(check_dir) if isinstance(check_dir, str) else check_dir
        if check_path.exists():
            jars = list(check_path.rglob("*.jar"))
            all_jar_locations.extend(jars)

    return all_jar_locations


def validate_tool(tool: Any) -> None:
    """Run quick validation test on tool."""
    try:
        quick_test = tool.check(VALIDATION_TEST_TEXT)
        if quick_test:
            print(
                f"✅ Quick validation: LanguageTool is working ({len(quick_test)} issue(s) detected in test)"
            )
        else:
            print("⚠️  Quick validation: No issues detected (may be normal)")
    except Exception as e:
        print(f"⚠️  Quick validation failed: {e}")


def get_languagetool_tool(language: str = "en-GB") -> Any:
    """Get or create LanguageTool tool instance."""
    global lt_tool

    if lt_tool is not None:
        print(f"📦 Using existing LanguageTool instance for {language}")
        return lt_tool

    try:
        start_time = time.time()
        print(f"🚀 Initializing LanguageTool for language: {language}...")

        cache_base = LT_CACHE_DIR
        jar_dir = LT_JAR_DIR
        os.makedirs(cache_base, exist_ok=True, mode=DIRECTORY_MODE)
        os.makedirs(jar_dir, exist_ok=True, mode=DIRECTORY_MODE)

        original_home, original_xdg_cache, original_ltp_path = setup_environment(
            cache_base, jar_dir
        )

        jar_exists, jar_location = find_jar_in_cache(jar_dir, cache_base)
        if jar_exists and jar_location:
            jar_size_mb = _bytes_to_mb(jar_location.stat().st_size)
            print(f"📦 Found cached JAR in volume: {jar_location} ({jar_size_mb:.1f}MB)")
        else:
            print("📥 JAR not found in cache, will download")

        ngram_path = setup_ngram_data(language)
        configure_ngrams(ngram_path)

        lt_tool = create_tool_with_config(language, ngram_path)

        all_jar_locations = _find_all_jar_locations(jar_dir, cache_base)

        if all_jar_locations:
            print(f"💾 Found {len(all_jar_locations)} JAR file(s):")
            for jar_file in all_jar_locations[:MAX_JAR_FILES_TO_SHOW]:
                jar_size_mb = _bytes_to_mb(jar_file.stat().st_size)
                print(f"   - {jar_file} ({jar_size_mb:.1f}MB)")

        copy_jar_to_volume(all_jar_locations, jar_dir)

        restore_environment(original_home, original_xdg_cache, original_ltp_path)

        init_time = time.time() - start_time
        print(f"✅ LanguageTool initialized successfully in {init_time:.2f}s")
        print(f"📊 LanguageTool instance: {type(lt_tool).__name__}")

        validate_tool(lt_tool)

        return lt_tool
    except Exception as e:
        error_msg = f"Failed to initialize LanguageTool: {str(e)}"
        print(f"❌ {error_msg}", file=sys.stderr)
        print(f"📜 Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        raise RuntimeError(error_msg) from None
