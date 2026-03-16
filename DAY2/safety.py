"""
Step 12 – Safety and Validation.

Provides two functions used at the start of every pipeline run:

validate_input(text)
    Checks the raw user input for:
    - empty / whitespace-only strings
    - inputs that exceed the maximum allowed length
    - patterns indicative of injection attacks (SQL injection, XSS,
      shell commands)

sanitize_input(text)
    Strips non-printable control characters and normalises whitespace.
    Should be applied after validate_input passes.

Design notes
------------
- Validation raises no exceptions; it returns (bool, reason) so the
  pipeline can decide how to respond.
- The blocked patterns target common injection vectors. They are not
  meant to be an exhaustive security layer but a basic first gate.
- Input length is capped to prevent prompt-stuffing attacks where an
  adversary embeds malicious instructions inside a very long message.
"""

import re

MAX_INPUT_LENGTH: int = 500

# Patterns for common injection vectors checked case-insensitively.
_BLOCKED_PATTERNS: list[str] = [
    # SQL injection
    r"\b(?:drop\s+table|delete\s+from|insert\s+into|select\s+\*|union\s+select)\b",
    # XSS / HTML injection
    r"<\s*script|javascript\s*:|on\w+\s*=",
    # Shell command injection
    r"(?:&&|\|\|)\s*\w+",
    r"\b(?:rm\s+-rf|format\s+c:|shutdown\s+/[rsh]|del\s+/[fsq])\b",
    # Prompt injection markers (instruct the model to ignore prior instructions)
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
]

_COMPILED: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in _BLOCKED_PATTERNS
]


def validate_input(text: str) -> tuple[bool, str]:
    """
    Step 12 – Input Validation.

    Args:
        text: Raw user input string.

    Returns:
        (is_safe, reason)
        is_safe – True when input passes all checks.
        reason  – Empty string on success; human-readable explanation on failure.
    """
    if not text or not text.strip():
        return False, "Input is empty."

    if len(text) > MAX_INPUT_LENGTH:
        return False, f"Input too long (max {MAX_INPUT_LENGTH} characters)."

    for pattern in _COMPILED:
        if pattern.search(text):
            return False, "Input contains disallowed content."

    return True, ""


def sanitize_input(text: str) -> str:
    """
    Step 12 – Input Sanitisation.

    Removes non-printable ASCII control characters (except tab and newline)
    and normalises runs of whitespace to a single space.

    Args:
        text: Raw user input string.

    Returns:
        Cleaned string safe to pass into downstream NLP functions.
    """
    # Remove non-printable characters except common whitespace
    text = re.sub(r"[^\x09\x0A\x20-\x7E]", "", text)
    # Collapse internal whitespace (preserve single spaces)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
