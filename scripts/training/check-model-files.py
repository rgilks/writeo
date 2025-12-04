#!/usr/bin/env python3
"""
Check what files exist in the model directory on Modal.
"""

import modal

app = modal.App("check-model-files")

volume = modal.Volume.from_name("writeo-models", create_if_missing=True)


@app.function(volumes={"/vol/models": volume})
def list_model_files():
    """List files in the model directory."""
    from pathlib import Path

    model_dir = Path("/vol/models/corpus-trained-roberta")

    print(f"Checking directory: {model_dir}")
    print(f"Exists: {model_dir.exists()}")

    if model_dir.exists():
        print(f"\nFiles in {model_dir}:")
        for item in sorted(model_dir.iterdir()):
            size = item.stat().st_size if item.is_file() else "-"
            print(
                f"  {item.name:40s} {'file' if item.is_file() else 'dir':6s} {size if isinstance(size, str) else f'{size:,} bytes'}"
            )
    else:
        print("Directory does not exist!")

    return list(str(f) for f in model_dir.iterdir()) if model_dir.exists() else []


@app.local_entrypoint()
def main():
    """Run check."""
    files = list_model_files.remote()
    print(f"\nFound {len(files)} items")
