import os
import re
from pathlib import Path


def check_links(root_dir):
    root_path = Path(root_dir).resolve()
    broken_links = []

    # Find all markdown files
    md_files = [f for f in root_path.glob("**/*.md") if "node_modules" not in str(f)]

    print(f"Scanning {len(md_files)} markdown files...")

    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    for file_path in md_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Find all links
        for match in link_pattern.finditer(content):
            text, link = match.groups()

            # Skip external links and anchors
            if link.startswith(("http", "https", "mailto:", "#")):
                continue

            # Handle anchor only links in file
            if link.startswith("#"):
                continue

            # Strip anchor from link for file check
            link_clean = link.split("#")[0]
            if not link_clean:
                continue

            # Resolve path
            # If start with /, it's relative to root, else relative to file
            if link_clean.startswith("/"):
                target_path = root_path / link_clean.lstrip("/")
            else:
                target_path = (file_path.parent / link_clean).resolve()

            if not target_path.exists():
                broken_links.append(
                    {
                        "file": str(file_path.relative_to(root_path)),
                        "text": text,
                        "link": link,
                        "target": str(target_path),
                    }
                )

    return broken_links


if __name__ == "__main__":
    current_dir = os.getcwd()
    print(f"Checking links in {current_dir}")
    broken = check_links(current_dir)

    if broken:
        print(f"\nFound {len(broken)} broken links:")
        for item in broken:
            print(f"File: {item['file']}")
            print(f"  Link: [{item['text']}]({item['link']})")
            print(f"  Target: {item['target']}")
            print("-" * 20)
        exit(1)
    else:
        print("\nAll links logic valid!")
        exit(0)
