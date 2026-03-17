"""
Step 9 – Routing Logic.

Decides whether the system should execute a local tool or hand off to
the LLM for reasoning.

Decision rule
-------------
  tool   – intent maps to a known tool AND all required slots are filled
  llm    – everything else (open-ended questions, missing information,
            unknown intent)

This keeps the routing layer simple and decoupled from both the tool
system and the LLM client.
"""

# Intents that have a corresponding tool implementation
_TOOL_INTENTS: frozenset[str] = frozenset({"weather", "restaurant", "math", "joke", "time"})


def route(intent: str, slots_satisfied: bool) -> str:
    """
    Step 9 – Routing Decision.

    Args:
        intent:           Detected intent label (from Step 4).
        slots_satisfied:  True when all required slots are filled (Step 6).

    Returns:
        'tool' – run a local tool function
        'llm'  – send to LLM for reasoning / generation
    """
    if intent in _TOOL_INTENTS and slots_satisfied:
        return "tool"
    return "llm"
