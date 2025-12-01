"""N-gram data setup utilities."""

import contextlib
import os
import tempfile
import traceback
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from config import LT_NGRAM_DIR, NGRAM_BASE_URL

# Constants
BYTES_PER_MB = 1024 * 1024
PROGRESS_UPDATE_INTERVAL = 1000  # Update progress every N blocks
DIRECTORY_MODE = 0o755
EXPECTED_FILE_SIZE_GB = 8.35
EXPECTED_FILE_SIZE_GB_APPROX = 8

# N-gram file names
NGRAM_FILE_2015 = "ngrams-en-20150817.zip"
NGRAM_FILE_LATEST = "ngrams-en.zip"


def _bytes_to_mb(bytes_value: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_value / BYTES_PER_MB


def get_language_code(language: str) -> str:
    """Get language code for n-gram directory."""
    lang_code_map = {
        "en-GB": "en",
        "en-US": "en",
        "en": "en",
    }
    return lang_code_map.get(language, language.split("-")[0])


def check_ngram_exists(language: str) -> str | None:
    """Check if n-gram data already exists."""
    lang_code = get_language_code(language)
    ngram_lang_dir = Path(LT_NGRAM_DIR) / lang_code

    if ngram_lang_dir.exists() and any(ngram_lang_dir.iterdir()):
        print(f"✅ N-gram data already exists at: {ngram_lang_dir}")
        return str(LT_NGRAM_DIR)
    return None


def _create_progress_callback() -> Callable[[int, int, int], None]:
    """Create a progress callback function for download."""

    def show_progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) / total_size)
            downloaded_mb = _bytes_to_mb(block_num * block_size)
            total_mb = _bytes_to_mb(total_size)
            if block_num % PROGRESS_UPDATE_INTERVAL == 0:
                print(
                    f"   Download progress: {percent:.1f}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)",
                    end="\r",
                )

    return show_progress


def _get_ngram_urls() -> list[str]:
    """Get list of possible n-gram download URLs to try."""
    return [
        f"{NGRAM_BASE_URL}{NGRAM_FILE_2015}",
        f"{NGRAM_BASE_URL}{NGRAM_FILE_LATEST}",
    ]


def _cleanup_temp_file(file_path: str | None) -> None:
    """Clean up temporary file if it exists."""
    if file_path and os.path.exists(file_path):
        with contextlib.suppress(OSError):
            os.unlink(file_path)


def download_ngram_zip() -> str | None:
    """Download n-gram zip file."""
    possible_urls = _get_ngram_urls()
    tmp_path = None

    try:
        for url in possible_urls:
            try:
                print(f"   Trying URL: {url}")
                fd, tmp_path = tempfile.mkstemp(suffix=".zip")
                os.close(fd)

                print(f"   Downloading to temporary file: {tmp_path}")
                print(
                    f"   This is a large file (~{EXPECTED_FILE_SIZE_GB}GB) and may take 10-15 minutes..."
                )

                progress_callback = _create_progress_callback()
                urllib.request.urlretrieve(url, tmp_path, progress_callback)
                print()

                file_size_mb = _bytes_to_mb(os.path.getsize(tmp_path))
                print(f"   ✅ Downloaded {file_size_mb:.1f}MB")

                # Validate zip file
                with zipfile.ZipFile(tmp_path, "r") as test_zip:
                    test_zip.testzip()

                print(f"   ✅ Successfully downloaded from: {url}")
                return tmp_path
            except Exception as url_error:
                print(f"   ✗ Failed: {url_error}")
                _cleanup_temp_file(tmp_path)
                tmp_path = None
                continue
    finally:
        # Ensure cleanup on unexpected exit
        if tmp_path and not os.path.exists(tmp_path):
            tmp_path = None

    return None


def extract_ngram_data(zip_path: str, ngram_lang_dir: Path) -> bool:
    """Extract n-gram data from zip file."""
    try:
        print("   Extracting n-gram data (this may take a while)...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            print(f"   Found {len(file_list)} files in archive")
            zip_ref.extractall(ngram_lang_dir)

        _cleanup_temp_file(zip_path)

        extracted_files = list(ngram_lang_dir.iterdir())
        if extracted_files:
            print(f"✅ N-gram data extracted successfully to: {ngram_lang_dir}")
            print(f"   Found {len(extracted_files)} files/directories")
            return True
        else:
            print("⚠️  Warning: N-gram extraction completed but directory is empty")
            return False
    except Exception as e:
        print(f"⚠️  Error extracting n-gram data: {e}")
        return False


def setup_ngram_data(language: str = "en-GB") -> str | None:
    """Download and set up n-gram data for LanguageTool."""
    existing = check_ngram_exists(language)
    if existing:
        return existing

    lang_code = get_language_code(language)
    ngram_lang_dir = Path(LT_NGRAM_DIR) / lang_code

    print(f"📥 N-gram data not found. Setting up n-grams for {language}...")
    print(f"   Target directory: {ngram_lang_dir}")
    os.makedirs(ngram_lang_dir, exist_ok=True, mode=DIRECTORY_MODE)

    if lang_code != "en":
        return None

    print("📥 Downloading English n-gram data...")
    print(
        f"   This is a large download (~{EXPECTED_FILE_SIZE_GB_APPROX}GB) and may take several minutes."
    )

    try:
        zip_path = download_ngram_zip()
        if not zip_path:
            raise RuntimeError("Could not download from any URL")

        if extract_ngram_data(zip_path, ngram_lang_dir):
            return str(LT_NGRAM_DIR)

        return None
    except Exception as e:
        print(f"⚠️  Could not download n-gram data automatically: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        print("   N-grams will be disabled. To enable manually:")
        print(f"   1. Visit: {NGRAM_BASE_URL}")
        print(f"   2. Download the English n-gram data (~{EXPECTED_FILE_SIZE_GB_APPROX}GB)")
        print(f"   3. Extract to: {ngram_lang_dir}")
        print("   4. Restart the service")
        return None
