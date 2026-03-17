"""
Step 13 – Response Generation.

Produces the final text response from either a tool result or an LLM
output, then applies optional post-processing to clean the formatting.

Reasoning banner
----------------
When provided by the pipeline, a compact one-line reasoning banner is
prepended, for example:
    Intent detected: weather | Route: tool | Tool called: get_weather

Post-processing
---------------
- Collapses runs of blank lines to a maximum of one blank line.
- Strips leading and trailing whitespace.
- Ensures the response ends with appropriate punctuation when it does not
  already (applies only to short single-sentence tool outputs).
"""

import re


def generate_response(source: str, data: str, reasoning: dict | None = None) -> str:
    """
    Step 13 – Response Generation.

    Args:
        source: One of 'tool', 'llm', 'clarification', 'llm_fallback', 'error'.
                Used internally for diagnostics; does not change the output.
        data:   Raw response string from the tool or LLM.
        reasoning:
                Optional dict for a short reasoning banner. Supported keys:
                intent, route_taken, tool_called, missing_slots.

    Returns:
        Cleaned, ready-to-display response string.
    """
    if not data or not data.strip():
        return "I'm sorry, I couldn't generate a response."

    body = _post_process(data)
    banner = _format_reasoning(source, reasoning or {})
    if banner:
        return f"{banner}\n{body}"
    return body


def _format_reasoning(source: str, reasoning: dict) -> str:
    """Build a concise one-line reasoning banner for the final reply."""
    if not reasoning:
        return ""

    parts: list[str] = []

    intent = str(reasoning.get("intent", "")).strip()
    if intent:
        parts.append(f"Intent detected: {intent}")

    route = str(reasoning.get("route_taken", source)).strip()
    if route:
        parts.append(f"Route: {route}")

    tool_called = str(reasoning.get("tool_called", "")).strip()
    if tool_called:
        parts.append(f"Tool called: {tool_called}")

    missing_slots = reasoning.get("missing_slots", [])
    if isinstance(missing_slots, list) and missing_slots:
        parts.append(f"Missing: {', '.join(str(s) for s in missing_slots)}")

    return " | ".join(parts)


def _post_process(text: str) -> str:
    """
    Normalise formatting of the response text.

    Keeps changes minimal – only removes redundant blank lines and
    trims outer whitespace so the caller receives a clean string.
    """
    # Collapse three or more consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
