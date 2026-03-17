"""
Step 11 – Tool System.

Defines the available tools, validates their arguments, and executes them.

Tools
-----
get_weather(city)     – return a weather description for the city
                        (OpenWeatherMap if WEATHER_API_KEY is set, else wttr.in)
get_restaurants(city) – find nearby restaurants using Geoapify
add_numbers(a, b)     – add two numbers and return a formatted result
tell_joke()           – generate and return a fresh joke via the LLM
get_time()            – return the current local time

Security
--------
Only tools listed in ALLOWED_TOOLS can be called.  Any attempt to
invoke an unlisted tool is rejected before execution.

Argument validation is performed before each tool call.  Invalid
arguments produce a descriptive error string rather than a crash.
"""

from __future__ import annotations

import os
from datetime import datetime

import requests

from DAY2.llm_client import ask_llm

# ---------------------------------------------------------------------------
# Joke history – prevents repeats within a session
# ---------------------------------------------------------------------------
_joke_history: list[str] = []
_MAX_JOKE_HISTORY = 10


# ---------------------------------------------------------------------------
# Allowed tools (whitelist)
# ---------------------------------------------------------------------------
ALLOWED_TOOLS: frozenset[str] = frozenset(
    {"get_weather", "add_numbers", "tell_joke", "get_time", "get_restaurants"}
)

# Map intent labels → tool names
_INTENT_TO_TOOL: dict[str, str] = {
    "weather":    "get_weather",
    "restaurant": "get_restaurants",
    "math":       "add_numbers",
    "joke":       "tell_joke",
    "time":       "get_time",
}


def get_tool_name_for_intent(intent: str) -> str | None:
    """Return the tool name mapped to an intent, if any."""
    return _INTENT_TO_TOOL.get(intent)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def _validate_args(tool_name: str, entities: dict) -> tuple[bool, str]:
    """
    Check that `entities` contains the arguments required by `tool_name`.

    Returns (is_valid, error_message).
    error_message is empty on success.
    """
    if tool_name == "get_weather":
        city = entities.get("city", "")
        if not city or not str(city).strip():
            return False, "City is required for weather lookup."

    elif tool_name == "get_restaurants":
        city = entities.get("city", "")
        if not city or not str(city).strip():
            return False, "City is required for restaurant search."

    elif tool_name == "add_numbers":
        nums = entities.get("numbers", [])
        if not isinstance(nums, list) or len(nums) < 2:
            return False, "Two numbers are required for addition."
        try:
            float(nums[0])
            float(nums[1])
        except (TypeError, ValueError):
            return False, "Numbers must be numeric values."

    return True, ""


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """
    Fetch real-time weather for `city`.

    Uses OpenWeatherMap when the WEATHER_API_KEY environment variable is set
    (free API key at openweathermap.org).  Falls back to the keyless wttr.in
    public API so the feature works out of the box without any configuration.
    """
    owm_key = os.environ.get("WEATHER_API_KEY", "")
    if owm_key:
        return _get_weather_owm(city, owm_key)
    return _get_weather_wttr(city)


def _get_weather_owm(city: str, api_key: str) -> str:
    """Fetch weather from the OpenWeatherMap API (requires API key)."""
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={requests.utils.quote(city)}&appid={api_key}&units=metric"
        )
        resp = requests.get(url, timeout=8)
        data = resp.json()

        if data.get("cod") != 200:
            msg = data.get("message", "unknown error")
            return f"Weather data for {city} could not be retrieved ({msg})."

        temp_c    = data["main"]["temp"]
        feels_c   = data["main"]["feels_like"]
        humidity  = data["main"]["humidity"]
        wind_kmph = round(data["wind"]["speed"] * 3.6, 1)
        desc      = data["weather"][0]["description"]

        return (
            f"Weather in {city}: {desc}. "
            f"Temperature {temp_c}°C (feels like {feels_c}°C), "
            f"humidity {humidity}%, wind {wind_kmph} km/h."
        )
    except requests.exceptions.ConnectionError:
        return "Could not reach the weather service. Check your internet connection."
    except requests.exceptions.Timeout:
        return f"Weather request for {city} timed out. Please try again."
    except (KeyError, ValueError, requests.exceptions.RequestException) as exc:
        return f"Weather data for {city} could not be retrieved ({exc})."


def _get_weather_wttr(city: str) -> str:
    """Fetch weather from wttr.in (no API key required)."""
    try:
        url = f"https://wttr.in/{requests.utils.quote(city)}?format=j1"
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()

        # wttr.in may return either {"current_condition": ...}
        # or {"data": {"current_condition": ...}} depending on upstream changes.
        payload = data.get("data", data)
        if not isinstance(payload, dict):
            return f"Weather data for {city} could not be retrieved (unexpected payload format)."

        conditions = payload.get("current_condition")
        if not isinstance(conditions, list) or not conditions:
            return f"Weather data for {city} could not be retrieved (missing current conditions)."

        cond = conditions[0]

        desc_block = cond.get("weatherDesc", [])
        if isinstance(desc_block, list) and desc_block and isinstance(desc_block[0], dict):
            desc = desc_block[0].get("value", "Unknown")
        else:
            desc = "Unknown"

        temp_c = cond.get("temp_C", "?")
        feels_c = cond.get("FeelsLikeC", "?")
        humidity = cond.get("humidity", "?")
        wind_kmph = cond.get("windspeedKmph", "?")

        return (
            f"Weather in {city}: {desc}. "
            f"Temperature {temp_c}°C (feels like {feels_c}°C), "
            f"humidity {humidity}%, wind {wind_kmph} km/h."
        )
    except requests.exceptions.ConnectionError:
        return "Could not reach the weather service. Check your internet connection."
    except requests.exceptions.Timeout:
        return f"Weather request for {city} timed out. Please try again."
    except (KeyError, ValueError, requests.exceptions.RequestException) as exc:
        return f"Weather data for {city} could not be retrieved ({exc})."


def add_numbers(a: float | int, b: float | int) -> str:
    """Add two numbers and return a human-readable result."""
    result = a + b
    # Format without decimal point when both operands are whole numbers
    if float(a) == int(float(a)) and float(b) == int(float(b)):
        return f"{int(float(a))} + {int(float(b))} = {int(float(result))}"
    return f"{a} + {b} = {result}"


def tell_joke() -> str:
    """
    Generate a fresh, non-repeating joke via the LLM.

    Maintains a session history of the last 10 jokes and passes them
    to the LLM so it cannot repeat them.  Format is restricted to
    classic setups ('Why did...', 'What do you call...') to avoid
    long-winded story-style jokes.
    """
    global _joke_history

    history_block = ""
    if _joke_history:
        history_block = (
            "Do NOT repeat or paraphrase any of these recent jokes:\n"
            + "\n".join(f"- {j}" for j in _joke_history)
            + "\n\n"
        )

    token = datetime.now().isoformat()
    prompt = (
        f"{history_block}"
        "Generate one short, original joke.\n"
        "Preferred formats: 'Why did...', 'What do you call...', "
        "'How many X does it take...', 'What\'s the difference between...'\n"
        "Rules:\n"
        "- Maximum two lines (setup + punchline).\n"
        "- Do NOT start with 'I tried', 'I told', 'I explained', or 'I asked'.\n"
        "- Clean and family-friendly.\n"
        f"Uniqueness token: {token}\n"
        "Return only the joke text, nothing else."
    )
    response = ask_llm(prompt).strip()
    if response and not response.lower().startswith(("llm", "error")):
        _joke_history.append(response)
        if len(_joke_history) > _MAX_JOKE_HISTORY:
            _joke_history.pop(0)
        return response
    return "Why did the scarecrow win an award? Because he was outstanding in his field."


def get_time() -> str:
    """Return the current local time as a formatted string."""
    return f"Current time is {datetime.now().strftime('%H:%M:%S')}."


def get_restaurants(city: str) -> str:
    """
    Find nearby restaurants using the Geoapify Places API.

    Two-step process:
    1. Geocode the city name to lat/lon coordinates.
    2. Search for restaurants within a 2 km radius.

    Returns a formatted list of up to 5 restaurant names.
    """
    api_key = os.environ.get("GEOAPIFY_KEY", "")
    if not api_key:
        return "Restaurant search is not configured (missing GEOAPIFY_KEY)."

    try:
        # Step 1: Geocode city → lat/lon
        geo_url = (
            f"https://api.geoapify.com/v1/geocode/search"
            f"?text={requests.utils.quote(city)}&apiKey={api_key}"
        )
        geo_resp = requests.get(geo_url, timeout=8)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data.get("features"):
            return f"Could not find location '{city}'. Please check the city name."

        props = geo_data["features"][0]["properties"]
        lat = props["lat"]
        lon = props["lon"]

        # Step 2: Search for restaurants near the coordinates
        place_url = (
            f"https://api.geoapify.com/v2/places"
            f"?categories=catering.restaurant"
            f"&filter=circle:{lon},{lat},2000"
            f"&limit=5&apiKey={api_key}"
        )
        place_resp = requests.get(place_url, timeout=8)
        place_resp.raise_for_status()
        place_data = place_resp.json()

        names = [
            feat["properties"].get("name", "Unnamed")
            for feat in place_data.get("features", [])
            if feat["properties"].get("name")
        ]

        if names:
            return f"Top restaurants in {city}: {', '.join(names)}."
        return f"No restaurants found near {city}."

    except requests.exceptions.ConnectionError:
        return "Could not reach the restaurant service. Check your internet connection."
    except requests.exceptions.Timeout:
        return f"Restaurant search for {city} timed out. Please try again."
    except (KeyError, ValueError, requests.exceptions.RequestException) as exc:
        return f"Restaurant data for {city} could not be retrieved ({exc})."


# ---------------------------------------------------------------------------
# Unified execution entry point
# ---------------------------------------------------------------------------

def execute_tool(intent: str, entities: dict) -> tuple[str | None, str]:
    """
    Step 11 – Tool Execution.

    Selects the appropriate tool for `intent`, validates arguments, then
    runs the tool.

    Args:
        intent:   Intent label from Step 4.
        entities: Extracted entities from Step 5 (after context resolution).

    Returns:
        (result, error)
        result – tool output string on success, None on failure
        error  – empty string on success, descriptive message on failure
    """
    tool_name = _INTENT_TO_TOOL.get(intent)
    if not tool_name:
        return None, f"No tool registered for intent '{intent}'."

    if tool_name not in ALLOWED_TOOLS:
        return None, f"Tool '{tool_name}' is not in the allowed tools list."

    valid, err = _validate_args(tool_name, entities)
    if not valid:
        return None, err

    if tool_name == "get_weather":
        city = str(entities["city"]).strip().title()
        return get_weather(city), ""

    if tool_name == "get_restaurants":
        city = str(entities["city"]).strip().title()
        return get_restaurants(city), ""

    if tool_name == "add_numbers":
        nums = entities["numbers"]
        return add_numbers(nums[0], nums[1]), ""

    if tool_name == "tell_joke":
        return tell_joke(), ""

    if tool_name == "get_time":
        return get_time(), ""

    return None, "Unexpected tool execution failure."
