"""
Step 3 – Preprocessing.

Cleans and normalises user input before further processing.

Pipeline:
    raw text
    → lowercase
    → strip punctuation / extra whitespace
    → tokenise
    → remove stopwords
    → return both original and normalised forms
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def _ensure_nltk() -> None:
    """Download required NLTK resources if they are not present."""
    for resource_path, package in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                # punkt_tab may not exist in older NLTK — fall back to punkt
                if package == "punkt_tab":
                    try:
                        nltk.data.find("tokenizers/punkt")
                    except LookupError:
                        nltk.download("punkt", quiet=True)


def preprocess(text: str) -> dict:
    """
    Step 3 – Preprocessing.

    Args:
        text: Raw user input string.

    Returns:
        dict with keys:
            original        – unmodified input
            normalized      – lowercased, punctuation-stripped text
            tokens          – word_tokenize result on normalised text
            filtered_tokens – tokens with stopwords removed
    """
    _ensure_nltk()

    original = text

    # Lowercase and normalise punctuation
    normalized = text.lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized)   # remove punctuation
    normalized = re.sub(r"\s+", " ", normalized).strip()

    tokens = word_tokenize(normalized)

    stop_words = set(stopwords.words("english"))
    filtered_tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    return {
        "original": original,
        "normalized": normalized,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
    }
