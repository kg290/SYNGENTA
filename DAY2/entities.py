"""
Step 5 – Entity Extraction.

Extracts structured information from raw user text using four methods:

1. Regex extraction       – pattern-based number and city detection
2. Dictionary matching    – lookup against a known-city vocabulary
3. spaCy NER              – catches arbitrary city names not in the dictionary
4. LLM extraction         – Gemini fills in gaps when the above fail

The public function `extract_entities` merges all four sources, only
calling the LLM when cheaper methods have not yet satisfied the need.
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Lazy spaCy loader – gracefully disabled if not installed
# ---------------------------------------------------------------------------

_nlp = None
_spacy_available: bool | None = None  # None = not yet probed


def _get_nlp():
    """Return a loaded spaCy model, or None if spaCy / the model is absent."""
    global _nlp, _spacy_available
    if _spacy_available is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            _spacy_available = True
        except Exception:
            _spacy_available = False
    return _nlp if _spacy_available else None


# ---------------------------------------------------------------------------
# Known-city vocabulary for dictionary matching
# ---------------------------------------------------------------------------
KNOWN_CITIES: set[str] = {
    "tokyo", "delhi", "mumbai", "london", "paris", "berlin", "beijing",
    "sydney", "moscow", "cairo", "new york", "chicago", "los angeles",
    "toronto", "dubai", "singapore", "bangkok", "seoul", "rome", "madrid",
    "amsterdam", "vienna", "zurich", "istanbul", "jakarta", "nairobi",
    "accra", "lagos", "johannesburg", "riyadh", "tehran", "warsaw",
    "budapest", "stockholm", "oslo", "helsinki", "brussels", "lisbon",
    "athens", "prague", "dublin", "edinburgh", "miami", "houston",
    "san francisco", "seattle", "boston", "montreal", "vancouver",
    "melbourne", "auckland", "cape town", "casablanca",
    # Indian cities
    "pune", "hyderabad", "bangalore", "bengaluru", "chennai", "kolkata",
    "jaipur", "ahmedabad", "lucknow", "chandigarh", "indore", "bhopal",
    "nagpur", "kochi", "coimbatore", "visakhapatnam", "surat", "vadodara",
    "goa", "mysore", "mangalore", "thiruvananthapuram", "guwahati",
    "patna", "ranchi", "dehradun", "shimla", "amritsar", "noida", "gurgaon",
}


# ---------------------------------------------------------------------------
# Method 1: Regex extraction
# ---------------------------------------------------------------------------

# Words that may follow "in/for/at" but are NOT city names
_CITY_STOPWORDS: set[str] = {
    "lunch", "dinner", "breakfast", "brunch", "food", "eating",
    "me", "today", "tomorrow", "now", "general", "that", "this",
    "morning", "afternoon", "evening", "night", "restaurants",
    "a", "an", "the", "my", "your", "it", "there", "here",
}


def extract_entities_regex(text: str) -> dict:
    """
    Extract entities using hand-written regex patterns.

    Finds:
        numbers – all integers and floats in the text
        city    – a word or phrase preceded by 'in', 'for', or 'at'
                  (case-insensitive; skips common non-city stopwords)
    """
    entities: dict = {}

    # Numbers (integers and decimals, including negatives)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        entities["numbers"] = [
            float(n) if "." in n else int(n) for n in numbers
        ]

    # City pattern: "in Tokyo" / "for New York" / "at berlin"
    # Case-insensitive; iterates through all matches to skip stopwords.
    for match in re.finditer(
        r"\b(?:in|for|at)\s+([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+)?)",
        text,
    ):
        candidate = match.group(1)
        if candidate.lower() not in _CITY_STOPWORDS:
            entities["city"] = candidate.title()
            break

    return entities


# ---------------------------------------------------------------------------
# Method 2: Dictionary matching
# ---------------------------------------------------------------------------

def extract_entities_dict(text: str) -> dict:
    """
    Match the text against a vocabulary of known city names.

    Case-insensitive; restores proper casing from the original text.
    Longer city names are checked before shorter ones to avoid partial
    matches (e.g. 'New York' before 'York').  Word boundaries are
    enforced so 'goa' does not match inside 'goat'.
    """
    entities: dict = {}
    lower = text.lower()

    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        idx = lower.find(city)
        if idx != -1:
            end_idx = idx + len(city)
            # Enforce word boundaries to avoid substring false positives
            before_ok = (idx == 0 or not lower[idx - 1].isalpha())
            after_ok = (end_idx >= len(lower) or not lower[end_idx].isalpha())
            if before_ok and after_ok:
                entities["city"] = text[idx: end_idx].title()
                break

    return entities


# ---------------------------------------------------------------------------
# Method 2b: spaCy NER extraction
# ---------------------------------------------------------------------------

def extract_entities_spacy(text: str) -> dict:
    """
    Use spaCy Named Entity Recognition to find location entities (GPE, LOC).

    Catches city names that are absent from KNOWN_CITIES and don't follow
    the regex "in/for/at City" pattern.  Runs locally – no API cost.
    Returns an empty dict when spaCy or its model is unavailable.
    """
    nlp = _get_nlp()
    if nlp is None:
        return {}
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            city = ent.text.strip("?.!,").title()
            if city:
                return {"city": city}
    return {}


# ---------------------------------------------------------------------------
# Helper: city name sanity check
# ---------------------------------------------------------------------------

def _is_plausible_city(name: str) -> bool:
    """
    Return True when `name` looks like a real city name.

    Rejects error messages returned by the LLM when no city was found,
    e.g. "I could not generate a response."
    """
    if not name:
        return False
    words = name.strip().split()
    if len(words) > 4 or len(name) > 50:
        return False
    # Every word must be alphabetic (hyphens and apostrophes allowed)
    return all(re.match(r"^[a-zA-Z\-']+$", w) for w in words)


# ---------------------------------------------------------------------------
# Method 3: LLM extraction
# ---------------------------------------------------------------------------

def extract_entities_llm(text: str, intent: str) -> dict:
    """
    Use the LLM to extract entities when regex and dictionary matching failed.

    Only called when intent-specific slots are still missing, to minimise
    unnecessary API calls.
    """
    from DAY2.llm_client import ask_llm

    if intent == "weather":
        prompt = (
            f"Extract the city name from this sentence: '{text}'\n"
            "Return only the city name. If none is present, return an empty string."
        )
        raw = ask_llm(prompt).strip().strip('"').strip("'")
        city = raw.splitlines()[0].strip() if raw else ""
        if _is_plausible_city(city) and not city.lower().startswith(("llm", "error", "i ", "none", "n/a")):
            return {"city": city.title()}
        return {}

    if intent == "restaurant":
        prompt = (
            f"Extract the city name from this sentence: '{text}'\n"
            "Return only the city name. If none is present, return an empty string."
        )
        raw = ask_llm(prompt).strip().strip('"').strip("'")
        city = raw.splitlines()[0].strip() if raw else ""
        if _is_plausible_city(city) and not city.lower().startswith(("llm", "error", "i ", "none", "n/a")):
            return {"city": city.title()}
        return {}

    if intent == "math":
        prompt = (
            f"Extract exactly two numbers from: '{text}'\n"
            'Return JSON only: {"a": <first number>, "b": <second number>}'
        )
        raw = ask_llm(prompt).strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                a = data.get("a", 0)
                b = data.get("b", 0)
                return {"numbers": [float(a) if "." in str(a) else int(a),
                                    float(b) if "." in str(b) else int(b)]}
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return {}

    return {}


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def extract_entities(text: str, intent: str) -> dict:
    """
    Step 5 – Entity Extraction.

    Merges regex → dictionary → spaCy NER → LLM results.
    Each subsequent method only fills slots that the previous one missed.

    Args:
        text:   Raw or lightly cleaned user input.
        intent: Detected intent label (guides which entities to look for).

    Returns:
        dict potentially containing:
            city    – string city name (for weather / restaurant)
            numbers – list of two numbers (for math)
    """
    entities: dict = {}

    # --- Regex ---
    entities.update(extract_entities_regex(text))

    # --- Dictionary (fills only what regex missed) ---
    for key, value in extract_entities_dict(text).items():
        if key not in entities:
            entities[key] = value

    # --- spaCy NER (fills city if regex+dict missed it; local, no API cost) ---
    if "city" not in entities:
        for key, value in extract_entities_spacy(text).items():
            if key not in entities:
                entities[key] = value

    # --- LLM (only called when required slots are still empty AND there is
    #     sufficient content in the input to plausibly contain a city) ---
    if intent == "weather" and "city" not in entities:
        # Skip LLM when the input is just the intent keyword with no extra tokens
        extra_tokens = [
            w for w in re.findall(r"\b[a-z]+\b", text.lower())
            if w not in {"weather", "the", "is", "in", "for", "at", "a", "an",
                         "what", "tell", "me", "please", "how", "check"}
        ]
        if extra_tokens:
            entities.update(extract_entities_llm(text, intent))

    if intent == "restaurant" and "city" not in entities:
        extra_tokens = [
            w for w in re.findall(r"\b[a-z]+\b", text.lower())
            if w not in {"restaurant", "restaurants", "find", "best", "places",
                         "to", "eat", "food", "near", "me", "the", "is", "in",
                         "for", "at", "a", "an", "where", "can", "i", "dining",
                         "lunch", "dinner", "breakfast", "brunch", "cafe"}
        ]
        if extra_tokens:
            entities.update(extract_entities_llm(text, intent))

    if intent == "math" and (
        "numbers" not in entities or len(entities.get("numbers", [])) < 2
    ):
        llm_result = extract_entities_llm(text, intent)
        if "numbers" in llm_result:
            entities["numbers"] = llm_result["numbers"]

    return entities
