#!/usr/bin/env python3
"""
Check what the trainer actually saved.
"""

import modal

app = modal.App("check-trainer-output")

volume = modal.Volume.from_name("writeo-models", create_if_missing=True)


@app.function(volumes={"/vol/models": volume})
def list_all_files():
    """List ALL files in /vol/models."""
    import os
    from pathlib import Path

    models_dir = Path("/vol/models")

    print(f"Contents of {models_dir}:")
    print("=" * 80)

    if not models_dir.exists():
        print("Directory doesn't exist!")
        return []

    all_items = []
    for root, dirs, files in os.walk(models_dir):
        level = root.replace(str(models_dir), "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            size = os.path.getsize(os.path.join(root, file))
            print(f"{subindent}{file} ({size:,} bytes)")
            all_items.append(os.path.join(root, file))

    return all_items


@app.local_entrypoint()
def main():
    """Run check."""
    files = list_all_files.remote()
    print(f"\n\nTotal files: {len(files)}")
