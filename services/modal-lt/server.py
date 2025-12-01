"""LanguageTool server management utilities."""

import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subprocess import Popen

# Constants
DEFAULT_PORT = 8081
JAVA_MEMORY_MB = 1536  # 1.5GB heap size for LanguageTool
SERVER_STARTUP_WAIT_SECONDS = 2  # Time to wait for server to start
ERROR_OUTPUT_MAX_LENGTH = 200  # Max characters to show from error output


def _build_java_command(jar_path: str, port: int, ngram_path: str | None = None) -> list[str]:
    """Build Java command for starting LanguageTool server."""
    java_cmd = [
        "java",
        f"-Xmx{JAVA_MEMORY_MB}m",
        "-jar",
        jar_path,
        "--port",
        str(port),
    ]

    if ngram_path:
        java_cmd.extend(["--languageModel", ngram_path])

    return java_cmd


def _check_server_started(server_process: "Popen[str]", port: int) -> bool:
    """Check if server process started successfully."""
    time.sleep(SERVER_STARTUP_WAIT_SECONDS)

    if server_process.poll() is None:
        print(f"✅ LanguageTool server started on port {port}")
        return True
    else:
        stdout, stderr = server_process.communicate()
        print("❌ LanguageTool server failed to start")
        if stdout:
            print(f"   stdout: {stdout[:ERROR_OUTPUT_MAX_LENGTH]}")
        if stderr:
            print(f"   stderr: {stderr[:ERROR_OUTPUT_MAX_LENGTH]}")
        return False


def start_languagetool_server_with_ngrams(
    jar_path: str, ngram_path: str | None = None, port: int = DEFAULT_PORT
) -> "Popen[str] | None":
    """Start LanguageTool Java server with optional n-gram support.

    Args:
        jar_path: Path to LanguageTool JAR file.
        ngram_path: Optional path to n-gram data directory. If None, server starts without n-grams.
        port: Port number for the server (default: 8081).

    Returns:
        Popen process object if server started successfully, None otherwise.
    """
    try:
        java_cmd = _build_java_command(jar_path, port, ngram_path)

        if ngram_path:
            print(f"🚀 Starting LanguageTool server with n-grams: {ngram_path}")
        else:
            print("🚀 Starting LanguageTool server without n-grams")

        server_process = subprocess.Popen(
            java_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if _check_server_started(server_process, port):
            return server_process
        return None

    except FileNotFoundError as e:
        print(f"⚠️  Could not find Java executable: {e}")
        return None
    except subprocess.SubprocessError as e:
        print(f"⚠️  Subprocess error starting LanguageTool server: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Could not start LanguageTool server: {e}")
        return None
