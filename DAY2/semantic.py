"""
Step 7 – Semantic Understanding.

Provides two capabilities:

1. Embedding similarity
   Uses sentence-transformers (all-MiniLM-L6-v2) when available, and
   falls back to cosine-on-bag-of-words otherwise.  Used to match an
   ambiguous input against intent exemplars when rule + ML detection
   both return 'unknown'.

2. Query rewriting
   Very short inputs (≤3 tokens) are expanded by the LLM into a full
   question before the rest of the pipeline runs.  This improves entity
   extraction and intent detection on under-specified queries.

Both features are optional: the pipeline degrades gracefully if
sentence-transformers is not installed or the LLM is unavailable.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Encoder loader – lazy initialisation with graceful fallback
# ---------------------------------------------------------------------------

_encoder = None
_use_transformers: bool | None = None  # None = not yet determined


def _get_encoder():
    global _encoder, _use_transformers
    if _use_transformers is None:
        try:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer("all-MiniLM-L6-v2")
            _use_transformers = True
        except Exception:
            _use_transformers = False
    return _encoder if _use_transformers else None


# ---------------------------------------------------------------------------
# Lightweight BOW cosine used when sentence-transformers is absent
# ---------------------------------------------------------------------------

def _bow_vector(text: str) -> dict[str, int]:
    tokens = re.findall(r"[a-z']+", text.lower())
    vec: dict[str, int] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def _bow_cosine(a: str, b: str) -> float:
    va, vb = _bow_vector(a), _bow_vector(b)
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    norm_a = sum(v ** 2 for v in va.values()) ** 0.5
    norm_b = sum(v ** 2 for v in vb.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Public: similarity
# ---------------------------------------------------------------------------

def similarity(text_a: str, text_b: str) -> float:
    """
    Cosine similarity between two text strings.

    Uses sentence-transformers embeddings when available,
    bag-of-words cosine otherwise.

    Returns a float in [0, 1].
    """
    encoder = _get_encoder()
    if encoder is not None:
        import numpy as np
        vecs = encoder.encode([text_a, text_b], convert_to_numpy=True)
        a, b = vecs[0], vecs[1]
        denom = float((a @ a) ** 0.5 * (b @ b) ** 0.5)
        return float((a @ b) / denom) if denom else 0.0
    return _bow_cosine(text_a, text_b)


# ---------------------------------------------------------------------------
# Intent exemplar bank – used for semantic intent matching
# ---------------------------------------------------------------------------

INTENT_EXEMPLARS: dict[str, list[str]] = {
    "weather": [
        "weather forecast",
        "temperature outside",
        "is it raining",
        "sunny today",
        "weather report",
    ],
    "math": [
        "add two numbers",
        "calculate the sum",
        "arithmetic computation",
        "multiply numbers",
        "what is the total",
    ],
    "joke": [
        "tell me a joke",
        "make me laugh",
        "funny story",
        "humour me",
    ],
    "time": [
        "what time is it",
        "current time",
        "clock reading",
        "what hour",
    ],
    "restaurant": [
        "find restaurants nearby",
        "places to eat",
        "food near me",
        "best dining options",
        "where to eat",
        "lunch places nearby",
        "dinner options in the city",
        "breakfast spots",
        "places for lunch",
        "best cafes around",
    ],
}


def semantic_intent_match(text: str, threshold: float = 0.45) -> str | None:
    """
    Return the intent whose exemplars are most similar to `text`.

    Only returns a label when the best similarity score exceeds `threshold`.
    Returns None when:
      - no intent clears the threshold, or
      - the input is a greeting / social phrase (to prevent false positives)

    Args:
        text:      User input (normalised).
        threshold: Minimum cosine similarity to accept a match (0.45).
    """
    if is_social(text.strip().lower()):
        return None
    best_score = 0.0
    best_intent: str | None = None

    for intent, exemplars in INTENT_EXEMPLARS.items():
        for exemplar in exemplars:
            score = similarity(text, exemplar)
            if score > best_score:
                best_score = score
                best_intent = intent

    return best_intent if best_score >= threshold else None


# ---------------------------------------------------------------------------
# Social / greeting vocabulary and patterns
# These inputs are never rewritten or matched against domain exemplars.
# ---------------------------------------------------------------------------
_SOCIAL_PHRASES: set[str] = {
    # single-word greetings
    "hi", "hello", "hey", "hiya", "howdy", "greetings", "sup",
    # farewells
    "bye", "goodbye", "cya", "later", "farewell", "adieu",
    # thanks
    "thanks", "thank you", "thx", "cheers", "ty", "thankyou",
    # filler / reactions
    "ok", "okay", "sure", "great", "nice", "cool", "good", "fine",
    "yes", "no", "nope", "yep", "yup", "nah",
    "lol", "haha", "lmao", "rofl", "hehe", "xd",
    "hmm", "umm", "uh", "uhh", "ah", "ahh", "oh", "ohh", "err",
    "huh", "what", "help", "please", "sorry",
    "alright", "right", "indeed", "exactly",
    # multi-word social phrases (exact)
    "see you", "not bad", "sounds good", "no problem", "no worries",
    "of course", "sure thing", "oh okay", "ah okay", "oh ok", "i see",
    "excuse me", "my bad", "got it", "i got it",
}

# Regex patterns for social multi-word phrases not covered by exact match.
_SOCIAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(hi|hello|hey|howdy|hiya)\s+(there|all|everyone|folks|friend|bot|again)$"),
    re.compile(r"^good\s+(morning|afternoon|evening|night|day|one)$"),
    re.compile(r"^how\s+are\s+(you|things|it going|you doing|you today)"),
    re.compile(r"^i\s+(am|m)\s+(fine|good|great|okay|ok|well|alright|not\s+bad)"),
    re.compile(r"^(not\s+bad|sounds?\s+(good|great|okay|fine)|all\s+good)$"),
    re.compile(r"^(um+|uh+|ah+|oh+|hmm+|err+)\s*$"),
    re.compile(r"^(nice\s+(to\s+meet\s+you|one|job|work)|good\s+(job|work|one))"),
    re.compile(r"^(that\s+(makes\s+sense|is\s+(great|cool|nice|good|helpful)))"),
    re.compile(r"^(no\s+worries|no\s+problem|never\s+mind)"),
    re.compile(r"^(ah\s+i\s+see|oh\s+i\s+see|oh\s+okay|ah\s+ok)"),
    re.compile(r"^thanks?\s+(a\s+lot|so\s+much|very\s+much)"),
    re.compile(r"^(bye+|goodbye+|see\s+you\s+(later|soon|around))"),
    re.compile(r"^(lol|haha+|lmao|rofl|hehe+|xd)\s*[!.]*$"),
]

# Identity question patterns – answered by the bot itself.
_IDENTITY_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(what|who)\s+are\s+you"),
    re.compile(r"^what\s+(can|do)\s+you\s+(do|help|know)"),
    re.compile(r"^are\s+you\s+(a\s+)?(bot|robot|ai|assistant|human|real|person)"),
    re.compile(r"^what\s+is\s+your\s+name"),
    re.compile(r"^(your\s+name|tell\s+me\s+about\s+yourself|introduce\s+yourself)"),
    re.compile(r"^how\s+(do\s+you\s+work|can\s+you\s+help)"),
]


def is_social(text: str) -> bool:
    """Return True when the input is a greeting, filler, or social phrase."""
    clean = text.strip().lower()
    if clean in _SOCIAL_PHRASES:
        return True
    return any(p.search(clean) for p in _SOCIAL_PATTERNS)


def is_identity_question(text: str) -> bool:
    """Return True when the user is asking about the bot itself."""
    clean = text.strip().lower()
    return any(p.search(clean) for p in _IDENTITY_PATTERNS)


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

def rewrite_query(text: str) -> str:
    """
    Expand a very short, domain-related query into a complete question.

    Only rewrites when:
      - input has 3 or fewer tokens
      - input is NOT a greeting/social phrase
      - input contains at least one domain-related keyword

    Returns the original text unchanged in all other cases.

    Examples:
        "weather?"     → "What is the current weather forecast?"
        "Tokyo"        → "What is the weather in Tokyo?"
        "hi"           → "hi"  (unchanged – social phrase)
        "joke"         → "Can you tell me a joke?"  (domain keyword)
    """
    clean = text.strip().lower()
    tokens = clean.split()

    if len(tokens) > 3:
        return text

    # Never rewrite greetings or social filler
    if is_social(clean):
        return text

    # Only rewrite if at least one domain hint is present
    _DOMAIN_HINTS = {
        "weather", "rain", "temperature", "forecast", "sunny", "snow", "wind",
        "add", "sum", "plus", "total", "calculate", "math",
        "joke", "funny", "laugh", "humor", "humour",
        "time", "clock", "hour",
        "restaurant", "restaurants", "food", "eat", "dining", "cuisine",
        "lunch", "dinner", "breakfast", "brunch", "cafe",
    }
    # Also rewrite bare city names (single capitalised word that isn't social)
    # and single domain words like "joke", "time"
    has_domain_hint = any(t in _DOMAIN_HINTS for t in tokens)
    if not has_domain_hint:
        return text

    from DAY2.llm_client import ask_llm

    prompt = (
        f"The user typed a short query: '{text}'\n"
        "It appears to be an incomplete domain query (weather / restaurant / math / joke / time).\n"
        "Rewrite it as one clear, complete question.\n"
        "If it is a greeting or unrelated to those topics, return it unchanged.\n"
        "Return only the rewritten question, nothing else."
    )
    expanded = ask_llm(prompt).strip()
    if expanded and not expanded.lower().startswith(("llm", "error")):
        return expanded
    return text
