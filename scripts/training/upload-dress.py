"""
Upload DREsS dataset to Modal volume for training.
"""

import modal

# Volume for training data
data_volume = modal.Volume.from_name("writeo-training-data", create_if_missing=True)

app = modal.App("writeo-upload-dress")


@app.function(
    volumes={"/data": data_volume},
    timeout=600,
)
def upload_dress_from_local(dress_files: dict[str, bytes]):
    """Upload DREsS files to Modal volume."""
    import os

    dress_dir = "/data/dress"
    os.makedirs(dress_dir, exist_ok=True)

    for filename, content in dress_files.items():
        filepath = os.path.join(dress_dir, filename)
        with open(filepath, "wb") as f:
            f.write(content)
        print(f"✅ Uploaded {filename} ({len(content)} bytes)")

    # List uploaded files
    print("\n📁 DREsS directory contents:")
    for f in os.listdir(dress_dir):
        size = os.path.getsize(os.path.join(dress_dir, f))
        print(f"   {f}: {size:,} bytes")

    data_volume.commit()
    return {"status": "success", "files": list(dress_files.keys())}


@app.local_entrypoint()
def main():
    """Upload DREsS dataset files."""
    from pathlib import Path

    # DREsS files to upload
    dress_path = Path.home() / "Desktop" / "DREsS"

    if not dress_path.exists():
        print(f"❌ DREsS folder not found at {dress_path}")
        return

    # Collect files
    files_to_upload = {}
    target_files = ["DREsS_Std.tsv", "DREsS_New.tsv"]

    for filename in target_files:
        filepath = dress_path / filename
        if filepath.exists():
            print(f"📦 Reading {filename}...")
            with open(filepath, "rb") as f:
                files_to_upload[filename] = f.read()
        else:
            print(f"⚠️ {filename} not found, skipping")

    if not files_to_upload:
        print("❌ No DREsS files found to upload")
        return

    print(f"\n🚀 Uploading {len(files_to_upload)} files to Modal volume...")
    result = upload_dress_from_local.remote(files_to_upload)
    print(f"\n✅ Upload complete: {result}")


if __name__ == "__main__":
    main()
