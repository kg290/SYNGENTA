"""
Step 10 – LLM Integration.

Handles all Gemini API communication:
    - loading the API key from .env
    - sending prompts
    - receiving responses
    - error handling

All other modules that need the LLM import from here.
"""

import os

from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client():
    """
    Return a cached Gemini client, or None when GEMINI_API_KEY is not set.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        from google import genai
        _client = genai.Client(api_key=api_key)
    return _client


def ask_llm(prompt: str) -> str:
    """
    Send a plain prompt to the LLM and return the response text.

    Returns a descriptive error string (never raises) so callers can
    degrade gracefully instead of crashing.
    """
    client = get_client()
    if client is None:
        return "LLM unavailable. Set GEMINI_API_KEY in .env to enable Gemini."
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text or "I could not generate a response."
    except Exception as exc:
        return f"LLM error: {exc}"


def ask_llm_reasoned(user_input: str) -> str:
    """
    Ask the LLM to reason about and answer a general user question.

    Uses a system prompt that instructs the model to be honest about
    uncertainty and to avoid fabricating facts.
    """
    prompt = (
        "You are a helpful, accurate assistant.\n"
        "Answer the user's question clearly and concisely.\n"
        "If uncertain, say so. Do not invent facts.\n\n"
        f"User question: {user_input}"
    )
    return ask_llm(prompt)
