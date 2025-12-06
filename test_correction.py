from services.modal_gec.correction import ErrorExtractor


def test_extractor():
    print("Testing ErrorExtractor...")
    extractor = ErrorExtractor(use_errant=False)  # Force fallback for local test

    source = "I has three book"
    target = "I have three books"

    edits = extractor.extract_edits(source, target)
    print(f"Source: {source}")
    print(f"Target: {target}")
    print("Edits:")
    for edit in edits:
        print(edit)

    assert len(edits) > 0
    print("Test passed!")


if __name__ == "__main__":
    test_extractor()
