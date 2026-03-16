"""
Step 5 – Entity Extraction.

Extracts structured information from raw user text using three methods:

1. Regex extraction       – pattern-based number and city detection
2. Dictionary matching    – lookup against a known-city vocabulary
3. LLM extraction         – Gemini fills in gaps when the above fail

The public function `extract_entities` merges all three sources, only
calling the LLM when cheaper methods have not yet satisfied the need.
"""

from __future__ import annotations

import json
import re

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
}


# ---------------------------------------------------------------------------
# Method 1: Regex extraction
# ---------------------------------------------------------------------------

def extract_entities_regex(text: str) -> dict:
    """
    Extract entities using hand-written regex patterns.

    Finds:
        numbers – all integers and floats in the text
        city    – a capitalised word or phrase preceded by 'in', 'for', or 'at'
    """
    entities: dict = {}

    # Numbers (integers and decimals, including negatives)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        entities["numbers"] = [
            float(n) if "." in n else int(n) for n in numbers
        ]

    # City pattern: "in Tokyo" / "for New York" / "at Berlin"
    city_match = re.search(
        r"\b(?:in|for|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text,
    )
    if city_match:
        entities["city"] = city_match.group(1)

    return entities


# ---------------------------------------------------------------------------
# Method 2: Dictionary matching
# ---------------------------------------------------------------------------

def extract_entities_dict(text: str) -> dict:
    """
    Match the text against a vocabulary of known city names.

    Case-insensitive; restores proper casing from the original text.
    Longer city names are checked before shorter ones to avoid partial
    matches (e.g. 'New York' before 'York').
    """
    entities: dict = {}
    lower = text.lower()

    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        idx = lower.find(city)
        if idx != -1:
            entities["city"] = text[idx: idx + len(city)].title()
            break

    return entities


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

    Merges regex → dictionary → LLM results.
    Each subsequent method only fills slots that the previous one missed.

    Args:
        text:   Raw or lightly cleaned user input.
        intent: Detected intent label (guides which entities to look for).

    Returns:
        dict potentially containing:
            city    – string city name (for weather)
            numbers – list of two numbers (for math)
    """
    entities: dict = {}

    # --- Regex ---
    entities.update(extract_entities_regex(text))

    # --- Dictionary (fills only what regex missed) ---
    for key, value in extract_entities_dict(text).items():
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

    if intent == "math" and (
        "numbers" not in entities or len(entities.get("numbers", [])) < 2
    ):
        llm_result = extract_entities_llm(text, intent)
        if "numbers" in llm_result:
            entities["numbers"] = llm_result["numbers"]

    return entities
