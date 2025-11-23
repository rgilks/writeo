"""N-gram data setup utilities."""

import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from config import LT_NGRAM_DIR, NGRAM_BASE_URL


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


def download_ngram_zip() -> str | None:
    """Download n-gram zip file."""
    possible_urls = [
        "https://languagetool.org/download/ngram-data/ngrams-en-20150817.zip",
        f"{NGRAM_BASE_URL}ngrams-en-20150817.zip",
        f"{NGRAM_BASE_URL}ngrams-en.zip",
    ]

    tmp_path = None
    for url in possible_urls:
        try:
            print(f"   Trying URL: {url}")
            fd, tmp_path = tempfile.mkstemp(suffix=".zip")
            os.close(fd)

            def show_progress(block_num: int, block_size: int, total_size: int) -> None:
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) / total_size)
                    downloaded_mb = (block_num * block_size) / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    if block_num % 1000 == 0:
                        print(
                            f"   Download progress: {percent:.1f}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)",
                            end="\r",
                        )

            print(f"   Downloading to temporary file: {tmp_path}")
            print("   This is a large file (~8.35GB) and may take 10-15 minutes...")
            urllib.request.urlretrieve(url, tmp_path, show_progress)
            print()
            file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            print(f"   ✅ Downloaded {file_size_mb:.1f}MB")

            with zipfile.ZipFile(tmp_path, "r") as test_zip:
                test_zip.testzip()

            print(f"   ✅ Successfully downloaded from: {url}")
            return tmp_path
        except Exception as url_error:
            print(f"   ✗ Failed: {url_error}")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                tmp_path = None
            continue

    return None


def extract_ngram_data(zip_path: str, ngram_lang_dir: Path) -> bool:
    """Extract n-gram data from zip file."""
    try:
        print("   Extracting n-gram data (this may take a while)...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            print(f"   Found {len(file_list)} files in archive")
            zip_ref.extractall(ngram_lang_dir)

        if zip_path and os.path.exists(zip_path):
            os.unlink(zip_path)

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
    os.makedirs(ngram_lang_dir, exist_ok=True, mode=0o755)

    if lang_code != "en":
        return None

    print("📥 Downloading English n-gram data...")
    print("   This is a large download (~8GB) and may take several minutes.")

    try:
        zip_path = download_ngram_zip()
        if not zip_path:
            raise Exception("Could not download from any URL")

        if extract_ngram_data(zip_path, ngram_lang_dir):
            return str(LT_NGRAM_DIR)

        return None
    except Exception as e:
        print(f"⚠️  Could not download n-gram data automatically: {e}")
        import traceback

        print(f"   Error details: {traceback.format_exc()}")
        print("   N-grams will be disabled. To enable manually:")
        print(f"   1. Visit: {NGRAM_BASE_URL}")
        print("   2. Download the English n-gram data (~8GB)")
        print(f"   3. Extract to: {ngram_lang_dir}")
        print("   4. Restart the service")
        return None
