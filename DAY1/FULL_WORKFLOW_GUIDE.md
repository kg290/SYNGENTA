# Full NLP + LLM Walkthrough (Implemented System + Code Snippets + FAQ)

This is the complete walkthrough of what we implemented in this project, how it works, how each command uses rule or ML or LLM, and how to explain it in a presentation.

Main code file:
- functions.py

-------------------------------------------------------------------------------

## 1) What Is Needed For NLP + LLM To Work

For a practical NLP + LLM system, these building blocks are needed:

1. Input processing
- Accept raw user text.
- Normalize text so downstream steps are more stable.

2. Intent detection
- Identify user goal: weather, math, joke, time, or unknown.
- Use one or more strategies: rule, ML, LLM, hybrid.

3. Entity extraction
- Convert language into structured values (example: numbers, city).

4. Tool/action execution
- Run deterministic functions for tasks (math, weather, joke generation).

5. LLM reasoning layer
- Handle open-ended queries and uncertain cases.

6. Router/orchestrator
- Decide path: direct NLP tool execution, LLM tool calling, or direct reasoned answer.

7. Reliability controls
- Confidence thresholds.
- Fallback behavior when extraction/classification fails.

8. Interaction loop
- Support command-based control (intent mode, debugging commands).

Core system view:

User input
-> preprocess
-> intent detection
-> entity extraction (if needed)
-> tool execution (if needed)
-> LLM reasoned response (if needed)

-------------------------------------------------------------------------------

## 2) What We Implemented (Exact)

Implemented in functions.py:

1. NLP preprocessing
- ensure_nltk_resources
- preprocess_text
- run_step1_preprocessing

2. Multi-strategy intent detection
- detect_intent_rule_based
- detect_intent_ml
- detect_intent_llm
- detect_intent (mode router)

3. ML intent model
- _intent_features
- INTENT_TRAINING_DATA
- get_intent_classifier
- detect_intent_ml_with_confidence

4. Entity extraction
- extract_numbers (regex)
- extract_city (LLM + fallback)

5. Tool layer
- get_weather
- add_numbers
- get_new_joke (LLM-generated joke)
- tell_joke (wrapper)

6. LLM integration
- get_gemini_client
- ask_llm
- ask_llm_reasoned

7. LLM tool-calling simulation
- choose_tool_with_llm
- _extract_json_object
- execute_tool_call

8. End-to-end orchestration
- run_step5_pipeline
- run_step8_hybrid_pipeline
- run_step10_agent
- connect_everything
- main

-------------------------------------------------------------------------------

## 3) Code Snippets (From Implemented File)

### 3.1 Preprocessing

```python
def preprocess_text(text):
    ensure_nltk_resources()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in tokens if word.lower() not in stop_words]

    return tokens, filtered
```

What it does:
- tokenizes text
- removes stopwords
- returns cleaned signal for later steps

### 3.2 Rule-based intent detection

```python
def detect_intent_rule_based(text):
    text = text.lower()

    if "weather" in text:
        return "weather"

    if "add" in text or "sum" in text or "plus" in text or "total" in text:
        return "math"

    if "time" in text or "clock" in text:
        return "time"

    return "unknown"
```

What it does:
- fast deterministic intent classification

### 3.3 ML intent detection

```python
def _intent_features(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    features = {f"contains({token})": True for token in tokens}
    features["has_number"] = bool(re.search(r"\d", text))
    features["token_count"] = len(tokens)
    return features


def detect_intent_ml(text):
    classifier = get_intent_classifier()
    distribution = classifier.prob_classify(_intent_features(text))
    predicted = distribution.max()

    if distribution.prob(predicted) < ML_INTENT_CONFIDENCE_THRESHOLD:
        return "unknown"

    return predicted
```

What it does:
- converts text into features
- predicts intent using Naive Bayes
- applies confidence threshold

### 3.4 LLM intent detection

```python
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
```

What it does:
- delegates intent decision to LLM
- normalizes output into allowed labels

### 3.5 Hybrid intent routing

```python
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

    rule_intent = detect_intent_rule_based(text)
    if rule_intent != "unknown":
        return rule_intent

    ml_intent, ml_confidence = detect_intent_ml_with_confidence(text)

    if ml_intent == "joke":
        if ml_confidence >= ML_JOKE_CONFIDENCE_THRESHOLD:
            llm_intent = detect_intent_llm(text)
            return llm_intent if llm_intent in INTENT_LABELS else "unknown"
        return "unknown"

    if ml_intent != "unknown" and ml_confidence >= ML_INTENT_CONFIDENCE_THRESHOLD:
        return ml_intent

    return detect_intent_llm(text)
```

What it does:
- combines rule speed, ML confidence, and LLM fallback

### 3.6 Entity extraction

```python
def extract_numbers(text):
    nums = re.findall(r"-?\d+", text)

    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])

    return None


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
```

What it does:
- numbers via regex
- city via LLM extraction with fallback

### 3.7 Joke generation (new implementation)

```python
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
```

What it does:
- directly requests fresh joke from LLM
- no local joke bank stored

### 3.8 LLM integration

```python
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
```

What it does:
- central Gemini call abstraction
- unified error/fallback behavior

### 3.9 Tool-calling simulation

```python
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
```

What it does:
- asks LLM to choose tool + arguments in structured JSON

### 3.10 Final orchestrator

```python
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
```

What it does:
- full decision engine for the assistant

-------------------------------------------------------------------------------

## 4) Command To Engine Mapping (Clear Runtime Behavior)

Session control commands:

1. intent-mode: rule
- all future intent detection uses rule-based function

2. intent-mode: ml
- all future intent detection uses ML classifier

3. intent-mode: llm
- all future intent detection uses LLM classifier

4. intent-mode: hybrid
- order used: rule -> ML -> LLM fallback

5. detect-intent: any sentence
- returns only detected intent based on current mode

6. preprocess: any sentence
- runs only preprocessing pipeline (no intent detection)

Normal user request handling:

1. math-like request (add 3 and 9)
- intent detected
- numbers extracted
- add_numbers called

2. weather-like request
- intent detected as weather
- city extracted
- get_weather called

3. joke request
- intent detected as joke
- get_new_joke called (LLM-generated joke)

4. unknown query
- tries tool selection path
- otherwise falls back to ask_llm_reasoned

-------------------------------------------------------------------------------

## 5) What We Have Explored So Far

1. NLP preprocessing
2. Rule-based intent classification
3. ML intent classification with Naive Bayes
4. LLM intent classification
5. Hybrid intent routing
6. Regex + LLM entity extraction
7. Tool execution architecture
8. LLM fallback reasoning
9. LLM JSON tool selection
10. End-to-end orchestration loop
11. Improving response quality over iterations
12. Direct LLM-based fresh joke generation

-------------------------------------------------------------------------------

## 6) How We Achieved This (Implementation Journey)

1. Built preprocessing first.
2. Added baseline rule intent detection.
3. Added regex extraction for numbers.
4. Added core tools (math/weather/joke).
5. Added Gemini integration and fallback paths.
6. Added ML intent classifier with starter dataset.
7. Added LLM intent classifier.
8. Combined all three in hybrid routing.
9. Added LLM tool-calling simulation via JSON.
10. Unified everything under run_step10_agent.
11. Added runtime commands for debugging and mode switching.
12. Updated joke generation to use direct LLM output (no local joke storage).

-------------------------------------------------------------------------------

## 7) Demo Run Commands

Run from DAY1 folder:

..\venv\Scripts\python.exe .\functions.py

Suggested demo flow:
1. intent-mode: hybrid
2. preprocess: I am learning NLP and LLM integration
3. detect-intent: tell me a joke
4. add 7 and 5
5. weather in Tokyo
6. tell me a joke
7. explain black holes simply
8. exit

-------------------------------------------------------------------------------

## 8) What More Can Be Explored Next

1. Better ML dataset and metrics.
2. Real weather API integration.
3. Strict JSON schema for tool calls.
4. Conversation memory module.
5. Safety checks and prompt injection defenses.
6. Latency and cost optimization.
7. FastAPI service + UI frontend.
8. Automated test suite and benchmark prompts.

-------------------------------------------------------------------------------

## 9) Common Questions (25) - Detailed Project Answers

1. Why was an ML classifier used in this project?
- Rule logic works only when users use expected keywords.
- In real usage, users phrase the same intent in many ways, and keywords are often missing.
- The Naive Bayes model learns token patterns from INTENT_TRAINING_DATA, so it can detect intent from wording style, not only fixed words.
- This is why queries like "make me laugh" can map to joke even when rule-based logic has no hardcoded joke keyword.

2. Why keep rule-based intent detection if ML and LLM are available?
- Rules are deterministic, fast, and almost free to run.
- For strong patterns like add/sum/plus, rule outputs are stable and predictable.
- Rule-first behavior also reduces unnecessary LLM calls, which helps with latency and API usage.
- In this architecture, rules act as a high-precision first filter.

3. Why add LLM-based intent detection?
- ML model quality depends on training data size and diversity.
- Your starter dataset is intentionally small for learning, so edge cases are expected.
- LLM intent classification handles paraphrases better and recovers when rule and ML are uncertain.
- In this project, LLM is used as the flexible fallback layer.

4. Why use hybrid mode as default?
- Hybrid combines the strengths of each approach in a practical order.
- Current order is rule -> ML (with confidence checks) -> LLM fallback.
- This gives a balance of speed, cost, robustness, and adaptability.
- It also reflects how many production assistants use multiple decision layers.

5. Why apply confidence thresholds to ML predictions?
- A classifier always returns some label, even when uncertain.
- Without thresholds, weak predictions create wrong routing and poor user answers.
- ML_INTENT_CONFIDENCE_THRESHOLD prevents low-confidence decisions from being treated as ground truth.
- Uncertain cases are escalated to LLM for safer interpretation.

6. Why special handling for joke in hybrid mode?
- Joke examples are stylistically broad and can be over-predicted by small datasets.
- The project adds ML_JOKE_CONFIDENCE_THRESHOLD plus optional LLM verification path.
- This reduces accidental joke classification for unrelated casual text.
- It is a targeted safeguard based on observed model behavior.

7. Why are numbers extracted with regex instead of LLM?
- Numeric extraction is deterministic and easy with regex patterns.
- Regex is faster, cheaper, and stable for arithmetic input formats.
- For this use case, regex provides enough precision without added LLM complexity.
- This is a good example of using simple tools for structured subproblems.

8. Why is city extraction done with LLM?
- City mentions can appear in flexible forms, not just "weather in X".
- LLM extraction handles natural phrasing and implicit context better.
- The prompt is constrained to return only the city name to reduce noise.
- A fallback heuristic remains in place when LLM is unavailable.

9. Why keep fallback logic in extract_city?
- External APIs can fail due to quota, network, or key issues.
- Without fallback, weather flow would completely break.
- The fallback keeps the system functional in degraded mode.
- This improves resilience and user experience.

10. Why use explicit tool functions instead of only LLM text answers?
- Tools give deterministic execution and testable behavior.
- add_numbers always performs real arithmetic instead of estimated text reasoning.
- Tool boundaries reduce hallucination risk for action-oriented tasks.
- This architecture cleanly separates reasoning from execution.

11. Why is get_weather mocked right now?
- This project focuses on NLP + LLM orchestration patterns first.
- Mocking weather keeps local development simple and deterministic.
- It avoids API integration overhead during early learning stages.
- The function signature is ready to swap to a real weather API later.

12. Why is ask_llm implemented as a central wrapper?
- A single wrapper standardizes all model calls across features.
- Error handling, model name, and response formatting are centralized.
- Future upgrades like retries, logging, and safety filters can be added once here.
- This prevents duplication and inconsistent model usage.

13. Why create ask_llm_reasoned as a separate function?
- Not all prompts should have the same behavior.
- ask_llm_reasoned adds instruction-level constraints: concise, accurate, uncertainty-aware.
- It is used for unknown/open-ended answers where quality and caution are important.
- This separation keeps intent/tool prompts clean and specialized.

14. Why force JSON in tool selection prompts?
- Tool execution needs structured, machine-readable outputs.
- JSON allows unambiguous parsing of selected tool and arguments.
- It makes validation easier before function calls.
- This mirrors modern function-calling patterns used in production systems.

15. What happens if JSON returned by LLM is malformed?
- _extract_json_object attempts direct parse, then tries object extraction from mixed text.
- If parsing still fails, choose_tool_with_llm returns a safe default tool none.
- The system then falls back to reasoned answering.
- This prevents crashes and unsafe execution.

16. Why call a final response prompt after tool execution?
- Tool outputs can be raw, short, or machine-like.
- Final prompting turns structured results into user-friendly language.
- The prompt also instructs grounding in tool output to avoid hallucination.
- This improves readability without losing factual anchor.

17. Why return joke tool results directly in the agent path?
- In earlier iterations, second-pass prompting paraphrased jokes awkwardly.
- Direct return preserves the generated joke quality and intent.
- It avoids unnecessary extra model call and latency for joke responses.
- This is a targeted optimization based on observed output behavior.

18. Why maintain intent-mode commands in the chat loop?
- intent-mode command allows live switching between rule, ml, llm, and hybrid.
- This makes comparison easy during demos and debugging.
- It also helps explain architecture behavior in presentations.
- You can demonstrate why one mode succeeds where another fails.

19. Why is detect-intent command useful?
- It isolates classification from full agent execution.
- You can test only intent routing without extraction/tool side effects.
- This is especially useful when tuning thresholds and training data.
- It provides transparent debugging for model decisions.

20. Why include preprocess command in user flow?
- preprocess command exposes tokenization and stopword removal outputs.
- It helps learners see NLP transformations directly.
- It verifies resources are installed and preprocessing works.
- It is a teaching and debugging command, not just a runtime feature.

21. Can the system run without GEMINI_API_KEY?
- Yes, partially.
- Rule-based and ML-based components still run.
- But LLM intent, city extraction, tool selection, and fresh joke generation degrade.
- In that state, some paths return explicit LLM-unavailable messages.

22. Why Naive Bayes and not deep learning for intent?
- Naive Bayes is lightweight, quick to train, and beginner-friendly.
- It works reasonably on small text datasets for prototype classification.
- Deep models require larger datasets, tuning effort, and more dependencies.
- For this phase, simplicity and explainability are the priority.

23. How can we quickly improve intent accuracy in this project?
- Expand INTENT_TRAINING_DATA with balanced, realistic phrases per intent.
- Add hard negatives that look similar but belong to different labels.
- Evaluate using train/test split and confusion matrix.
- Tune ML_INTENT_CONFIDENCE_THRESHOLD and ML_JOKE_CONFIDENCE_THRESHOLD from metrics.

24. How can tool calling be made safer here?
- Enforce strict schema validation on tool and argument types.
- Add numeric bounds and city text sanitization before execution.
- Reject unknown tools explicitly and log parsing errors.
- Add fallback retry prompt when JSON is malformed.

25. How close is this project to production architecture?
- Architecturally, it already includes key patterns: hybrid intent, extraction, tools, orchestration.
- For production, add: observability, rate limiting, retries, security checks, and automated tests.
- Replace mocked weather with real API integration and secrets management.
- Add evaluation pipelines so model and routing changes are measurable over time.

-------------------------------------------------------------------------------

## 10) Final Summary

This project now demonstrates a complete NLP + LLM workflow:
- preprocessing
- multi-strategy intent detection
- entity extraction
- deterministic tools
- LLM reasoning
- LLM tool calling
- agent orchestration

It is a strong base for both learning and production-oriented extension.
