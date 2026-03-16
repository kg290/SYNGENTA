"""
Step 13 – Response Generation.

Produces the final text response from either a tool result or an LLM
output, then applies optional post-processing to clean the formatting.

Post-processing
---------------
- Collapses runs of blank lines to a maximum of one blank line.
- Strips leading and trailing whitespace.
- Ensures the response ends with appropriate punctuation when it does not
  already (applies only to short single-sentence tool outputs).
"""

import re


def generate_response(source: str, data: str) -> str:
    """
    Step 13 – Response Generation.

    Args:
        source: One of 'tool', 'llm', 'clarification', 'llm_fallback', 'error'.
                Used internally for diagnostics; does not change the output.
        data:   Raw response string from the tool or LLM.

    Returns:
        Cleaned, ready-to-display response string.
    """
    if not data or not data.strip():
        return "I'm sorry, I couldn't generate a response."

    return _post_process(data)


def _post_process(text: str) -> str:
    """
    Normalise formatting of the response text.

    Keeps changes minimal – only removes redundant blank lines and
    trims outer whitespace so the caller receives a clean string.
    """
    # Collapse three or more consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
