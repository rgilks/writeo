#!/usr/bin/env python3
"""Quick script to list the exact volume contents."""

import modal

app = modal.App("check-volume")
volume = modal.Volume.from_name("writeo-models", create_if_missing=True)


@app.function(volumes={"/vol/models": volume})
def check():
    from pathlib import Path

    path = Path("/vol/models")
    print(f"Contents of {path}:")
    for item in sorted(path.iterdir()):
        print(f"  {item.name}")

    # Check corpus model specifically
    corpus_path = path / "corpus-trained-roberta"
    print(f"\nChecking {corpus_path}:")
    print(f"  Exists: {corpus_path.exists()}")
    if corpus_path.exists():
        print("  Files:")
        for f in sorted(corpus_path.iterdir()):
            print(f"    - {f.name}")


if __name__ == "__main__":
    with app.run():
        check.remote()
