"""LanguageTool tool loading utilities."""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from config import LT_CACHE_DIR, LT_JAR_DIR
from ngram_setup import setup_ngram_data

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


def configure_ngrams(ngram_path: str | None) -> None:
    """Configure n-gram environment variables."""
    if ngram_path:
        os.environ["LANGUAGETOOL_LANGUAGE_MODEL"] = ngram_path
        os.environ["LT_LANGUAGE_MODEL"] = ngram_path
        print(f"✅ Set LANGUAGETOOL_LANGUAGE_MODEL={ngram_path}")
        print(f"✅ Set LT_LANGUAGE_MODEL={ngram_path}")
    else:
        for env_var in ["LANGUAGETOOL_LANGUAGE_MODEL", "LT_LANGUAGE_MODEL"]:
            if env_var in os.environ:
                del os.environ[env_var]
        print("ℹ️  N-gram data not available - continuing without n-grams")


def create_tool_with_config(language: str, ngram_path: str | None) -> Any:
    """Create LanguageTool instance with configuration."""
    import language_tool_python

    test_grammar_text = "I goes to the store. He don't like it."
    print(f"🧪 Testing grammar detection with: '{test_grammar_text}'")

    tool_config: dict[str, Any] = {"maxSpellingSuggestions": 5}

    if ngram_path:
        tool_config["languageModel"] = ngram_path
        print(f"✅ Configuring LanguageTool with n-grams: {ngram_path}")
        print("   Note: n-gram path should point to parent directory (contains 'en/' folder)")

    test_tool = language_tool_python.LanguageTool(language, config=tool_config)
    test_matches = test_tool.check(test_grammar_text)
    print(f"🧪 Test grammar check found {len(test_matches)} issues")

    if test_matches:
        for match in test_matches[:3]:
            # Handle both camelCase and snake_case attribute names
            rule_id = getattr(match, "rule_id", getattr(match, "ruleId", "UNKNOWN"))
            category = getattr(match, "category", "UNKNOWN")
            message = getattr(match, "message", "")
            print(f"   - Rule: {rule_id}, Category: {category}, Message: {message[:50]}")
    else:
        print("⚠️  WARNING: No grammar errors detected in test text!")

    return test_tool


def copy_jar_to_volume(all_jar_locations: list[Path], jar_dir: str) -> None:
    """Copy JAR files to volume if not already there."""
    for jar_file in all_jar_locations:
        if str(jar_file).startswith("/vol"):
            continue
        else:
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


def validate_tool(tool: Any) -> None:
    """Run quick validation test on tool."""
    try:
        quick_test = tool.check("I goes to store")
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
        print(f"⏱️  Start time: {start_time:.2f}s")

        cache_base = LT_CACHE_DIR
        jar_dir = LT_JAR_DIR
        os.makedirs(cache_base, exist_ok=True, mode=0o755)
        os.makedirs(jar_dir, exist_ok=True, mode=0o755)

        original_home, original_xdg_cache, original_ltp_path = setup_environment(
            cache_base, jar_dir
        )

        jar_exists, jar_location = find_jar_in_cache(jar_dir, cache_base)
        if jar_exists and jar_location:
            jar_size_mb = jar_location.stat().st_size / 1024 / 1024
            print(f"📦 Found cached JAR in volume: {jar_location} ({jar_size_mb:.1f}MB)")
        else:
            print("📥 JAR not found in cache, will download")

        ngram_path = setup_ngram_data(language)
        configure_ngrams(ngram_path)

        lt_tool = create_tool_with_config(language, ngram_path)

        all_jar_locations = []
        for check_dir in [
            jar_dir,
            cache_base + "/.cache/language-tool-python",
            "/root/.cache/language-tool-python",
        ]:
            check_path = Path(check_dir)
            if check_path.exists():
                jars = list(check_path.rglob("*.jar"))
                all_jar_locations.extend(jars)

        if all_jar_locations:
            print(f"💾 Found {len(all_jar_locations)} JAR file(s):")
            for jar_file in all_jar_locations[:5]:
                jar_size_mb = jar_file.stat().st_size / 1024 / 1024
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
        import traceback

        print(f"📜 Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        raise RuntimeError(error_msg) from None
