"""LanguageTool server management utilities."""

import subprocess
from typing import Optional
from pathlib import Path


def start_languagetool_server_with_ngrams(
    jar_path: str, ngram_path: Optional[str] = None, port: int = 8081
) -> Optional[subprocess.Popen]:
    """Start LanguageTool Java server with n-gram support."""
    if not ngram_path:
        return None

    try:
        java_cmd = [
            "java",
            "-Xmx1536m",
            "-jar",
            jar_path,
            "--port",
            str(port),
        ]

        if ngram_path:
            java_cmd.extend(["--languageModel", ngram_path])
            print(f"🚀 Starting LanguageTool server with n-grams: {ngram_path}")
        else:
            print(f"🚀 Starting LanguageTool server without n-grams")

        server_process = subprocess.Popen(
            java_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        import time

        time.sleep(2)

        if server_process.poll() is None:
            print(f"✅ LanguageTool server started on port {port}")
            return server_process
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ LanguageTool server failed to start")
            print(f"   stdout: {stdout[:200]}")
            print(f"   stderr: {stderr[:200]}")
            return None

    except Exception as e:
        print(f"⚠️  Could not start LanguageTool server manually: {e}")
        return None

