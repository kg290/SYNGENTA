import json
import os
import re
from datetime import datetime

import nltk
from dotenv import load_dotenv
from google import genai
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


load_dotenv()
_gemini_client = None
_intent_classifier = None

INTENT_LABELS = {"weather", "math", "joke", "time", "unknown"}
INTENT_MODES = {"rule", "ml", "llm", "hybrid"}
ML_INTENT_CONFIDENCE_THRESHOLD = 0.45
ML_JOKE_CONFIDENCE_THRESHOLD = 0.70

# Small starter dataset for ML-based intent classification.
INTENT_TRAINING_DATA = [
    ("weather in tokyo", "weather"),
    ("what is the weather in delhi", "weather"),
    ("is it raining in mumbai", "weather"),
    ("temperature in london", "weather"),
    ("add 5 and 10", "math"),
    ("sum 12 and 8", "math"),
    ("what is 9 plus 4", "math"),
    ("calculate total of 100 and 50", "math"),
    ("tell me a joke", "joke"),
    ("make me laugh", "joke"),
    ("say something funny", "joke"),
    ("give me a funny line", "joke"),
    ("what time is it", "time"),
    ("tell me current time", "time"),
    ("show time now", "time"),
    ("current clock time", "time"),
    ("who won world cup", "unknown"),
    ("explain black holes", "unknown"),
    ("write a short poem", "unknown"),
    ("what is chlorophyll", "unknown"),
]


def ensure_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
    }

    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


def preprocess_text(text):
    ensure_nltk_resources()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in tokens if word.lower() not in stop_words]

    return tokens, filtered


def _intent_features(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    features = {f"contains({token})": True for token in tokens}
    features["has_number"] = bool(re.search(r"\d", text))
    features["token_count"] = len(tokens)
    return features


def get_intent_classifier():
    global _intent_classifier

    if _intent_classifier is None:
        training_set = [(_intent_features(text), label) for text, label in INTENT_TRAINING_DATA]
        _intent_classifier = NaiveBayesClassifier.train(training_set)

    return _intent_classifier


def detect_intent_rule_based(text):
    text = text.lower()

    if "weather" in text:
        return "weather"

    if "add" in text or "sum" in text or "plus" in text or "total" in text:
        return "math"

    if "time" in text or "clock" in text:
        return "time"

    return "unknown"


def detect_intent_ml(text):
    classifier = get_intent_classifier()
    distribution = classifier.prob_classify(_intent_features(text))
    predicted = distribution.max()

    if distribution.prob(predicted) < ML_INTENT_CONFIDENCE_THRESHOLD:
        return "unknown"

    return predicted


def detect_intent_ml_with_confidence(text):
    classifier = get_intent_classifier()
    distribution = classifier.prob_classify(_intent_features(text))
    predicted = distribution.max()
    confidence = distribution.prob(predicted)
    return predicted, confidence


def detect_intent_llm(text):
    prompt = f"""
Classify the user intent into exactly one label from this list:
weather, math, joke, time, unknown

User input: {text}

Return only the label, nothing else.
"""

    response = ask_llm(prompt).strip().lower()

    for label in INTENT_LABELS:
        if response == label or response.startswith(f"{label}\n"):
            return label

    if response.split():
        first_token = response.split()[0].strip(".,:;!?\"'()[]{}")
        if first_token in INTENT_LABELS:
            return first_token

    return "unknown"


def detect_intent(text, mode="hybrid"):
    mode = mode.lower().strip()

    if mode not in INTENT_MODES:
        mode = "hybrid"

    if mode == "rule":
        return detect_intent_rule_based(text)

    if mode == "ml":
        return detect_intent_ml(text)

    if mode == "llm":
        return detect_intent_llm(text)

    # Hybrid: rule for deterministic intents, then ML, then LLM fallback.
    rule_intent = detect_intent_rule_based(text)
    if rule_intent != "unknown":
        return rule_intent

    ml_intent, ml_confidence = detect_intent_ml_with_confidence(text)

    if ml_intent == "joke":
        # Joke can be over-predicted by tiny datasets, so verify with LLM first.
        if ml_confidence >= ML_JOKE_CONFIDENCE_THRESHOLD:
            llm_intent = detect_intent_llm(text)
            return llm_intent if llm_intent in INTENT_LABELS else "unknown"
        return "unknown"

    if ml_intent != "unknown" and ml_confidence >= ML_INTENT_CONFIDENCE_THRESHOLD:
        return ml_intent

    return detect_intent_llm(text)


def extract_numbers(text):
    nums = re.findall(r"-?\d+", text)

    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])

    return None


def get_weather(city):
    return f"Weather in {city} is sunny"


def add_numbers(a, b):
    return a + b


def get_new_joke(style_hint="general"):
    uniqueness_token = datetime.now().isoformat()

    prompt = """
Generate one fresh, clean one-sentence joke.
Style hint: {style_hint}
Uniqueness token: {uniqueness_token}
Return only the joke sentence.
""".format(style_hint=style_hint, uniqueness_token=uniqueness_token)

    response = ask_llm(prompt).strip()

    if response and not response.lower().startswith("llm unavailable") and not response.lower().startswith("llm error"):
        return response

    return "I cannot fetch a joke right now. Please check Gemini API access and try again."


def tell_joke():
    return get_new_joke()


def get_gemini_client():
    global _gemini_client

    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        _gemini_client = genai.Client(api_key=api_key)

    return _gemini_client


def ask_llm(prompt):
    client = get_gemini_client()

    if client is None:
        return "LLM unavailable. Set GEMINI_API_KEY in .env to enable Gemini."

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text or "I could not generate a response right now."
    except Exception as exc:
        return f"LLM error: {exc}"


def ask_llm_reasoned(user_input):
    prompt = f"""
You are a careful assistant.
Answer the user's question accurately and clearly.

Rules:
- Give the most correct answer you can.
- If uncertain, explicitly say what is uncertain.
- Keep the response concise and helpful.
- Do not invent tool outputs or fake facts.

User question: {user_input}
"""
    return ask_llm(prompt)


def extract_city(sentence):
    prompt = f"""
Extract the city name from this sentence.

Sentence: {sentence}

Return only the city name.
"""
    city = ask_llm(prompt).strip().strip('"').strip("'")
    city = city.splitlines()[0].strip() if city else ""

    if not city or city.lower().startswith("llm unavailable") or city.lower().startswith("llm error"):
        city = sentence.lower().replace("weather", "").replace("in", "").strip()

    return city.title() if city else "Your City"


def run_step1_preprocessing(text):
    tokens, filtered = preprocess_text(text)
    return {
        "tokens": tokens,
        "filtered": filtered,
        "normalized": " ".join(filtered).lower(),
    }


def run_step5_pipeline(user_input, intent_mode="hybrid"):
    intent = detect_intent(user_input, mode=intent_mode)

    if intent == "math":
        nums = extract_numbers(user_input)
        if nums:
            return str(add_numbers(nums[0], nums[1]))
        return "Please provide two numbers, for example: add 5 and 10."

    if intent == "joke":
        return tell_joke()

    if intent == "time":
        return f"Current time is {datetime.now().strftime('%H:%M:%S')}"

    return "I don't understand"


def run_step8_hybrid_pipeline(user_input):
    intent = detect_intent(user_input, mode="hybrid")

    if intent == "weather":
        city = extract_city(user_input)
        return get_weather(city)

    return ask_llm_reasoned(user_input)


def _extract_json_object(raw_text):
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def choose_tool_with_llm(user_input):
    tools = [
        "get_weather",
        "add_numbers",
        "tell_joke",
    ]

    prompt = f"""
You are a tool selection engine.
Available tools: {tools}

User request: {user_input}

Return ONLY valid JSON using this schema:
{{"tool": "get_weather|add_numbers|tell_joke|none", "arguments": {{"city": "", "a": 0, "b": 0}}}}

Rules:
- Use "none" if no tool is appropriate.
- For add_numbers, extract two numbers into a and b.
- For get_weather, extract city.
- Do not include any explanation.
"""

    raw_response = ask_llm(prompt)
    parsed = _extract_json_object(raw_response)

    if not isinstance(parsed, dict):
        return {"tool": "none", "arguments": {}}

    tool_name = parsed.get("tool", "none")
    arguments = parsed.get("arguments", {})

    if not isinstance(arguments, dict):
        arguments = {}

    return {
        "tool": tool_name,
        "arguments": arguments,
    }


def execute_tool_call(tool_call):
    tool_name = tool_call.get("tool", "none")
    arguments = tool_call.get("arguments", {})

    if tool_name == "tell_joke":
        return tell_joke()

    if tool_name == "get_weather":
        city = str(arguments.get("city", "Your City")).strip() or "Your City"
        return get_weather(city.title())

    if tool_name == "add_numbers":
        try:
            a = int(arguments.get("a"))
            b = int(arguments.get("b"))
            return str(add_numbers(a, b))
        except (TypeError, ValueError):
            return "I need two valid numbers to add."

    return None


def run_step10_agent(user_input, intent_mode="hybrid"):
    preprocessed = run_step1_preprocessing(user_input)
    normalized_text = preprocessed["normalized"] or user_input.lower()
    intent_source = user_input if intent_mode == "llm" else normalized_text
    intent = detect_intent(intent_source, mode=intent_mode)

    if intent in {"math", "joke", "time"}:
        return run_step5_pipeline(user_input, intent_mode=intent_mode)

    if intent == "weather":
        return run_step8_hybrid_pipeline(user_input)

    tool_call = choose_tool_with_llm(user_input)
    tool_result = execute_tool_call(tool_call)

    if tool_result:
        if tool_call.get("tool") == "tell_joke":
            return tool_result

        final_prompt = f"""
User request: {user_input}
Tool selected: {tool_call.get('tool')}
Tool output: {tool_result}

Respond to the user naturally in one short helpful sentence.
Ground your response only in the tool output above.
"""
        return ask_llm(final_prompt)

    return ask_llm_reasoned(user_input)


def connect_everything(user_input, intent_mode="hybrid"):
    return run_step10_agent(user_input, intent_mode=intent_mode)


def main():
    intent_mode = "hybrid"

    print("NLP + LLM Agent is ready. Type 'exit' to quit.")
    print("Try: preprocess: your sentence")
    print("Try: add 5 and 10 | tell me a joke | weather in Tokyo")
    print("Intent modes: rule | ml | llm | hybrid")
    print("Use: intent-mode: ml")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Goodbye")
            break

        if user_input.lower().startswith("preprocess:"):
            text = user_input.split(":", 1)[1].strip()
            preprocessed = run_step1_preprocessing(text)
            print("Tokens:", preprocessed["tokens"])
            print("Filtered:", preprocessed["filtered"])
            continue

        if user_input.lower().startswith("intent-mode:"):
            requested_mode = user_input.split(":", 1)[1].strip().lower()
            if requested_mode in INTENT_MODES:
                intent_mode = requested_mode
                print(f"Bot: Intent mode changed to '{intent_mode}'.")
            else:
                valid_modes = ", ".join(sorted(INTENT_MODES))
                print(f"Bot: Invalid mode. Choose one of: {valid_modes}")
            continue

        if user_input.lower().startswith("detect-intent:"):
            text = user_input.split(":", 1)[1].strip()
            print("Bot:", detect_intent(text, mode=intent_mode))
            continue

        print("Bot:", connect_everything(user_input, intent_mode=intent_mode))


if __name__ == "__main__":
    main()