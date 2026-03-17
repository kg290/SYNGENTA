"""
Step 4 – Intent Detection.

Implements four strategies and a hybrid fallback chain.

Strategies
----------
rule     – keyword regex matching (fast, deterministic)
ml       – Naive Bayes classifier trained on labelled examples
llm      – Gemini-based zero-shot classification
hybrid   – rule → ML → LLM fallback  (default)

Intents
-------
weather, math, joke, time, unknown
"""

from __future__ import annotations

import re

import nltk
from nltk.classify import NaiveBayesClassifier

INTENT_LABELS: set[str] = {"weather", "math", "joke", "time", "restaurant", "unknown"}
INTENT_MODES: set[str] = {"rule", "ml", "llm", "hybrid"}

# Common misspellings that should still map to restaurant intent.
_INTENT_TYPO_MAP: dict[str, str] = {
    "restraurant": "restaurant",
    "restraurants": "restaurants",
    "restraunt": "restaurant",
    "restraunts": "restaurants",
    "restuarant": "restaurant",
    "restuarants": "restaurants",
    "restarant": "restaurant",
    "restarants": "restaurants",
}

# Minimum probability an ML prediction must exceed to be accepted.
ML_CONFIDENCE_THRESHOLD: float = 0.45
# Joke intent requires a higher bar because it is easy to over-predict on
# small datasets.
JOKE_ML_CONFIDENCE_THRESHOLD: float = 0.70

# ---------------------------------------------------------------------------
# Labelled training data for the Naive Bayes classifier
# ---------------------------------------------------------------------------
TRAINING_DATA: list[tuple[str, str]] = [
    # weather
    ("weather in tokyo", "weather"),
    ("what is the weather in delhi", "weather"),
    ("is it raining in mumbai", "weather"),
    ("temperature in london", "weather"),
    ("forecast for paris", "weather"),
    ("how hot is it in berlin", "weather"),
    ("will it rain tomorrow", "weather"),
    ("sunny in sydney", "weather"),
    # math
    ("add 5 and 10", "math"),
    ("sum 12 and 8", "math"),
    ("what is 9 plus 4", "math"),
    ("calculate total of 100 and 50", "math"),
    ("7 times 6", "math"),
    ("multiply 3 by 4", "math"),
    ("divide 20 by 5", "math"),
    ("subtract 3 from 10", "math"),
    # joke
    ("tell me a joke", "joke"),
    ("make me laugh", "joke"),
    ("say something funny", "joke"),
    ("give me a funny one liner", "joke"),
    ("share a joke with me", "joke"),
    ("got any jokes", "joke"),
    # time
    ("what time is it", "time"),
    ("tell me the current time", "time"),
    ("show time now", "time"),
    ("current clock time", "time"),
    ("what hour is it", "time"),
    ("check the clock", "time"),
    # unknown
    ("who won the world cup", "unknown"),
    ("explain black holes", "unknown"),
    ("write a short poem", "unknown"),
    ("what is chlorophyll", "unknown"),
    ("tell me about history", "unknown"),
    ("how does photosynthesis work", "unknown"),
    # restaurant
    ("find restaurants near me", "restaurant"),
    ("food near me", "restaurant"),
    ("places to eat", "restaurant"),
    ("best restaurants in tokyo", "restaurant"),
    ("where can I eat", "restaurant"),
    ("restaurants in delhi", "restaurant"),
    ("dining options nearby", "restaurant"),
    ("recommend a restaurant", "restaurant"),
    ("places for lunch in pune", "restaurant"),
    ("lunch near me", "restaurant"),
    ("dinner places in paris", "restaurant"),
    ("best breakfast spots", "restaurant"),
    ("brunch options near me", "restaurant"),
    ("find a cafe in london", "restaurant"),
    ("where to get dinner", "restaurant"),
    ("lunch spots in bangalore", "restaurant"),
]

_classifier: NaiveBayesClassifier | None = None


# ---------------------------------------------------------------------------
# Feature extraction shared by training and inference
# ---------------------------------------------------------------------------

def _normalize_intent_text(text: str) -> str:
    """Normalise text and repair common domain typos for intent detection."""
    normalized = text.lower()
    for wrong, right in _INTENT_TYPO_MAP.items():
        normalized = re.sub(rf"\b{re.escape(wrong)}\b", right, normalized)
    return normalized

def _features(text: str) -> dict:
    text = _normalize_intent_text(text)
    tokens = re.findall(r"[a-z']+", text)
    feats: dict = {f"has({t})": True for t in tokens}
    feats["has_number"] = bool(re.search(r"\d", text))
    feats["token_count"] = len(tokens)
    return feats


def _get_classifier() -> NaiveBayesClassifier:
    global _classifier
    if _classifier is None:
        training_set = [(_features(t), lbl) for t, lbl in TRAINING_DATA]
        _classifier = NaiveBayesClassifier.train(training_set)
    return _classifier


# ---------------------------------------------------------------------------
# Individual detection strategies
# ---------------------------------------------------------------------------

def detect_intent_rule(text: str) -> str:
    """
    Rule-based intent detection using keyword regex.

    Fast and deterministic. Returns 'unknown' when no keywords match.
    Joke intent is intentionally excluded — it requires ML/LLM to avoid
    false positives on words like "funny" appearing in other contexts.
    """
    t = _normalize_intent_text(text)
    if re.search(r"\bweather\b|\brain\b|\bforecast\b|\btemperature\b|\bclimate\b|\bsunny\b|\bsnow\b", t):
        return "weather"
    if re.search(
        r"\brestaurant\b|\brestaurants\b|\bdining\b|\bcuisine\b"
        r"|\blunch\b|\bdinner\b|\bbreakfast\b|\bbrunch\b"
        r"|\bcafe\b|\bcafes\b|\beatery\b|\beateries\b"
        r"|\bplaces?\s+(?:to|for)\s+(?:eat|dine|lunch|dinner|breakfast)\b"
        r"|\bfood\s+(?:near|in|at|around)\b"
        r"|\bwhere\s+(?:can\s+i|to|should\s+i)\s+eat\b",
        t,
    ):
        return "restaurant"
    if re.search(r"\badd\b|\bsum\b|\bplus\b|\btotal\b|\btimes\b|\bmultiply\b|\bdivide\b|\bminus\b|\bsubtract\b|\bcalculate\b", t):
        return "math"
    if re.search(r"\btime\b|\bclock\b|\bhour\b|\bminute\b", t):
        return "time"
    return "unknown"


def detect_intent_ml(text: str) -> tuple[str, float]:
    """
    ML-based intent detection using a Naive Bayes classifier.

    Returns (intent_label, confidence).
    Low-confidence predictions are mapped to ('unknown', confidence).
    """
    clf = _get_classifier()
    dist = clf.prob_classify(_features(text))
    predicted = dist.max()
    confidence = dist.prob(predicted)
    if confidence < ML_CONFIDENCE_THRESHOLD:
        return "unknown", confidence
    return predicted, confidence


def detect_intent_llm(text: str) -> str:
    """
    LLM-based intent detection via zero-shot classification.

    Uses Gemini to classify the intent. Falls back to 'unknown' if the
    response cannot be parsed into a known label.
    """
    from DAY2.llm_client import ask_llm

    labels_str = ", ".join(sorted(INTENT_LABELS))
    prompt = (
        f"Classify the user intent into exactly one label from: {labels_str}\n"
        f"User input: {text}\n"
        "Return only the label, nothing else."
    )
    raw = ask_llm(prompt).strip().lower()

    for label in INTENT_LABELS:
        if raw == label or raw.startswith(label + "\n"):
            return label

    first = raw.split()[0].strip(".,;:!?\"'()") if raw.split() else ""
    if first in INTENT_LABELS:
        return first

    return "unknown"


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def detect_intent(text: str, mode: str = "hybrid") -> str:
    """
    Step 4 – Intent Detection.

    Args:
        text: Normalised user input text.
        mode: One of 'rule', 'ml', 'llm', 'hybrid' (default).

    Returns:
        Intent label string from INTENT_LABELS.

    Hybrid strategy:
        1. Rule-based  – returns immediately on a non-unknown result
        2. ML          – returns if confidence ≥ threshold (with extra check
                         for 'joke' to avoid over-triggering)
        3. LLM         – final fallback
    """
    mode = mode.lower().strip()
    if mode not in INTENT_MODES:
        mode = "hybrid"

    if mode == "rule":
        return detect_intent_rule(text)

    if mode == "ml":
        intent, _ = detect_intent_ml(text)
        return intent

    if mode == "llm":
        return detect_intent_llm(text)

    # --- hybrid ---
    rule_result = detect_intent_rule(text)
    if rule_result != "unknown":
        return rule_result

    ml_intent, ml_conf = detect_intent_ml(text)

    if ml_intent == "joke":
        # Require higher confidence for joke, then verify with LLM
        if ml_conf >= JOKE_ML_CONFIDENCE_THRESHOLD:
            return detect_intent_llm(text)
        return "unknown"

    if ml_intent != "unknown" and ml_conf >= ML_CONFIDENCE_THRESHOLD:
        return ml_intent

    return detect_intent_llm(text)
