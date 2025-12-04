#!/usr/bin/env python3
"""
Test Modal volume write and commit.
"""

import modal

app = modal.App("test-volume")

volume = modal.Volume.from_name("writeo-models", create_if_missing=True)


@app.function(volumes={"/vol/models": volume}, timeout=300)
def test_volume_write():
    """Test writing to volume and verify."""
    from pathlib import Path
    import json

    test_dir = Path("/vol/models/test-write")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Write test file
    test_file = test_dir / "test.txt"
    test_file.write_text("Hello from Modal!")

    # Write JSON
    json_file = test_dir / "config.json"
    json_file.write_text(json.dumps({"test": "data"}))

    print(f"✅ Wrote files to {test_dir}")
    print(f"   Files: {list(test_dir.iterdir())}")

    # Verify files exist before commit
    print("\n📋 Verification BEFORE commit:")
    for f in test_dir.iterdir():
        print(f"   {f.name}: {f.stat().st_size} bytes")

    # Explicit commit
    print("\n💾 Committing volume...")
    volume.commit()
    print("✅ Volume committed")

    return str(test_dir)


@app.function(volumes={"/vol/models": volume})
def verify_volume_write(test_dir_str: str):
    """Verify files exist after commit (in a new function call)."""
    from pathlib import Path

    test_dir = Path(test_dir_str)

    print("\n🔍 Verification AFTER commit (new function call):")
    print(f"   Directory exists: {test_dir.exists()}")

    if test_dir.exists():
        files = list(test_dir.iterdir())
        print(f"   Files found: {len(files)}")
        for f in files:
            print(f"   - {f.name}: {f.stat().st_size} bytes")
        return True
    else:
        print("   ❌ Directory not found!")
        return False


@app.local_entrypoint()
def main():
    """Run test."""
    print("=" * 80)
    print("TESTING MODAL VOLUME WRITE/COMMIT")
    print("=" * 80)

    # Write and commit
    test_dir = test_volume_write.remote()
    print(f"\nTest directory: {test_dir}")

    # Verify in new function (simulates model loading)
    success = verify_volume_write.remote(test_dir)

    print("\n" + "=" * 80)
    if success:
        print("✅ VOLUME WRITE/COMMIT WORKING!")
    else:
        print("❌ VOLUME WRITE/COMMIT FAILED!")
    print("=" * 80)
