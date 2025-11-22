#!/usr/bin/env python3
"""Remove secret from git history by rewriting commits."""

import subprocess
import sys

SECRET = "<YOUR-API-KEY>"
REPLACEMENT = "<YOUR-API-KEY>"

# Use git filter-repo if available (better than filter-branch)
try:
    result = subprocess.run(
        ["git", "filter-repo", "--replace-text", f"<(SECRET)={SECRET}=>{REPLACEMENT}>", "--force"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ Secret removed using git-filter-repo")
        sys.exit(0)
except FileNotFoundError:
    print("git-filter-repo not found, trying alternative method...")

# Alternative: Use BFG Repo-Cleaner if available
try:
    # Create a replacements file
    with open("/tmp/secret-replacements.txt", "w") as f:
        f.write(f"{SECRET}==>{REPLACEMENT}\n")
    
    result = subprocess.run(
        ["bfg", "--replace-text", "/tmp/secret-replacements.txt", "--no-blob-protection"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ Secret removed using BFG Repo-Cleaner")
        print("Run: git reflog expire --expire=now --all && git gc --prune=now --aggressive")
        sys.exit(0)
except FileNotFoundError:
    print("BFG Repo-Cleaner not found")

print("❌ Neither git-filter-repo nor BFG found.")
print("Install one of them:")
print("  pip install git-filter-repo")
print("  # or")
print("  brew install bfg")
print("\nOr manually use git filter-branch with proper sed syntax for your OS")

