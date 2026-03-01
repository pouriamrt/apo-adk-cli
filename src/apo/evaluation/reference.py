from __future__ import annotations

from difflib import SequenceMatcher


def reference_score(actual: str, expected: str) -> float:
    """Score actual output against expected output using fuzzy string matching."""
    actual = actual.strip()
    expected = expected.strip()

    # Exact match (case insensitive)
    if actual.lower() == expected.lower():
        return 1.0

    # Check if expected is contained in actual
    if expected.lower() in actual.lower():
        # Reward containment but penalize extra text
        ratio = len(expected) / max(len(actual), 1)
        return max(0.5, ratio)

    # Fuzzy string similarity
    return SequenceMatcher(None, actual.lower(), expected.lower()).ratio()
