"""
Step 6 – Slot Checking.

Verifies that all information required by a given intent is present in
the extracted entities.  When slots are missing, returns a human-readable
clarification message to send back to the user.

Slot requirements
-----------------
weather  → city (string, non-empty)
math     → numbers (list of at least two values)
joke     → (none)
time     → (none)
unknown  → (none)
"""

from __future__ import annotations

# Required slot names per intent
SLOT_REQUIREMENTS: dict[str, list[str]] = {
    "weather":    ["city"],
    "restaurant": ["city"],
    "math":       ["numbers"],
    "joke":       [],
    "time":       [],
    "unknown":    [],
}

# Human-readable prompt to send back when a slot is missing
CLARIFICATION_MESSAGES: dict[str, str] = {
    "city":    "Which city are you asking about?",
    "numbers": "Please provide two numbers. For example: add 5 and 10.",
}


def check_slots(intent: str, entities: dict) -> tuple[bool, list[str]]:
    """
    Step 6 – Slot Checking.

    Args:
        intent:   Detected intent label.
        entities: Extracted entity dict from Step 5.

    Returns:
        (all_satisfied, missing_slots)
        all_satisfied – True when every required slot is present and valid.
        missing_slots – list of slot names that are absent or invalid.
    """
    required = SLOT_REQUIREMENTS.get(intent, [])
    missing: list[str] = []

    for slot in required:
        if slot == "numbers":
            nums = entities.get("numbers", [])
            if not isinstance(nums, list) or len(nums) < 2:
                missing.append(slot)
        else:
            value = entities.get(slot)
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(slot)

    return (len(missing) == 0), missing


def clarification_prompt(missing_slots: list[str]) -> str:
    """
    Build a human-readable clarification request for any missing slots.

    Args:
        missing_slots: List of slot names returned by check_slots.

    Returns:
        A plain-text string to display to the user.
    """
    messages = [
        CLARIFICATION_MESSAGES.get(slot, f"Please provide: {slot}")
        for slot in missing_slots
    ]
    return " ".join(messages)
