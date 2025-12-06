import difflib
from typing import List, Dict, Any


class ErrorExtractor:
    def __init__(self, use_errant: bool = True):
        self.use_errant = use_errant
        self.annotator = None

        if self.use_errant:
            try:
                import errant
                import spacy

                # Load English annotator
                # Note: This assumes spacy model en_core_web_sm is installed
                try:
                    self.annotator = errant.load("en")
                except OSError:
                    print(
                        "Warning: Spacy model 'en' not found. Trying 'en_core_web_sm'."
                    )
                    try:
                        nlp = spacy.load("en_core_web_sm")
                        self.annotator = errant.load("en", nlp=nlp)
                    except Exception as e:
                        print(f"Warning: Could not load ERRANT annotator: {e}")
                        self.use_errant = False
            except ImportError:
                print(
                    "Warning: ERRANT or Spacy not installed. Falling back to simple diff."
                )
                self.use_errant = False

    def extract_edits(self, source: str, target: str) -> List[Dict[str, Any]]:
        """
        Extract edits between source and target text.
        """
        if self.use_errant and self.annotator:
            return self._extract_with_errant(source, target)
        else:
            return self._extract_with_difflib(source, target)

    def _extract_with_errant(self, source: str, target: str) -> List[Dict[str, Any]]:
        """Use ERRANT to extract and classify edits."""
        # Parse with spacy (via errant's annotator)
        src = self.annotator.parse(source)
        tgt = self.annotator.parse(target)

        # Annotate

        results = []
        for edit in self.annotator.annotate(src, tgt):
            # Convert ERRANT edit to our format
            results.append(
                {
                    "start": edit.o_start,
                    "end": edit.o_end,
                    "original": edit.o_str,
                    "correction": edit.c_str,
                    "type": self._map_errant_type(edit.type),
                    "errant_type": edit.type,
                }
            )

        return results

    def _extract_with_difflib(self, source: str, target: str) -> List[Dict[str, Any]]:
        """Fallback method using difflib."""
        source_words = source.split()
        target_words = target.split()
        matcher = difflib.SequenceMatcher(None, source_words, target_words)

        edits = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            original_text = " ".join(source_words[i1:i2])
            correction_text = " ".join(target_words[j1:j2])

            # Basic type inference for fallback
            error_type = "fluency"  # Default
            if tag == "delete":
                error_type = "grammar"  # Unnecessary word usually
            elif tag == "insert":
                error_type = "grammar"  # Missing word usually
            elif tag == "replace":
                # Heuristics
                if len(original_text) > 0 and len(correction_text) > 0:
                    # Check for simple capitalization or punctuation
                    if original_text.lower() == correction_text.lower():
                        error_type = "mechanics"

            edits.append(
                {
                    "start": i1,  # Word index
                    "end": i2,  # Word index
                    "original": original_text,
                    "correction": correction_text,
                    "type": error_type,
                    "tag": tag,
                }
            )

        return edits

    def _map_errant_type(self, errant_type: str) -> str:
        """Map ERRANT error types to our categories."""
        grammar_types = [
            "SVA",
            "VERB",
            "NOUN",
            "DET",
            "PREP",
            "ADJ",
            "ADV",
            "PRON",
            "PART",
            "MORPH",
        ]
        vocab_types = ["SPELL", "WO"]
        mechanics_types = ["PUNCT", "ORTH", "CAP"]

        if any(t in errant_type for t in grammar_types):
            return "grammar"
        elif any(t in errant_type for t in vocab_types):
            return "vocabulary"
        elif any(t in errant_type for t in mechanics_types):
            return "mechanics"
        else:
            return "fluency"
