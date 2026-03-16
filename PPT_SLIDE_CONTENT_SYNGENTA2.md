# SYNGENTA2 - PPT Slide Content (15 Slides)

Use this as direct slide text. Each slide has a title and concise talking points.

## Slide 1 - Title
**Building a Hybrid NLP + LLM Assistant (SYNGENTA2)**

- Objective: build a practical conversational assistant with deterministic tools + LLM reasoning.
- Evolution path: DAY1 monolithic prototype -> DAY2 modular production-style pipeline.
- Core principle: deterministic stages first, probabilistic stages next, generative fallback last.

## Slide 2 - Project Scope and Goals

- Support intent-driven tasks: weather, math, joke, time, and open-ended queries.
- Handle multi-turn conversations with context carry-over.
- Add safety checks before any NLP/LLM processing.
- Provide explainability via debug mode and internal stage visibility.

## Slide 3 - Architecture Overview (DAY2 Modules)

- safety.py: validation + sanitization gate.
- preprocessing.py + semantic.py: normalization, tokenization, rewrite, similarity fallback.
- intent.py + entities.py: intent classification and structured extraction.
- context.py + slots.py + routing.py: state, slot checks, decision control.
- tools.py + llm_client.py + response.py + pipeline.py + main.py: execution, generation, orchestration, REPL.

## Slide 4 - End-to-End Runtime Workflow

- Input -> validate_input -> sanitize_input.
- preprocess -> optional rewrite_query (short inputs) -> detect_intent.
- If unknown: semantic_intent_match fallback.
- extract_entities -> resolve_from_context -> check_slots.
- route -> execute_tool or ask_llm_reasoned.
- generate_response -> context.update.

## Slide 5 - Stage 1: Safety and Guardrails

- Threat checks: SQL injection, XSS/script injection, shell injection, prompt-injection patterns.
- Structural checks: empty input and max-length validation.
- sanitize_input removes non-printable characters and normalizes whitespace.
- Unsafe input is blocked early; downstream stages are not executed.

## Slide 6 - Stage 2: Preprocessing

- Normalization: lowercase + punctuation cleanup + whitespace normalization.
- Tokenization: splits text into tokens for downstream processing.
- Stopword filtering: keeps meaningful tokens for lightweight NLP signals.
- Output object retains original text, normalized text, tokens, and filtered tokens.

## Slide 7 - Stage 3: Semantic Enrichment

- Query rewriting for short inputs (<= 3 tokens) using LLM.
- Intent semantic fallback via exemplar similarity.
- Preferred similarity: sentence-transformers embeddings.
- Graceful degradation: bag-of-words cosine fallback if embeddings unavailable.

## Slide 8 - Stage 4: Intent Detection (4 Modes)

- Rule mode: fast deterministic keyword/regex intent mapping.
- ML mode: Naive Bayes classifier with confidence thresholding.
- LLM mode: zero-shot label classification.
- Hybrid mode: rule -> ML -> LLM fallback chain with stricter handling for noisy classes.

## Slide 9 - Stage 5: Entity Extraction (Layered)

- Regex extraction: numbers and city patterns (for math/weather).
- Dictionary extraction: known-city gazetteer fallback.
- LLM extraction: invoked only when required slots are unresolved.
- Merge strategy: regex -> dictionary -> LLM for cost and reliability balance.

## Slide 10 - Stage 6 and 7: Context + Slot Clarification

- DialogueContext stores last intent, last entities, unresolved slots, and recent turn history.
- Context resolution fills missing values from prior turns when valid.
- Slot schema enforces required arguments per intent.
- Missing slots trigger clarification flow (for example: "Which city are you asking about?").

## Slide 11 - Stage 8 and 9: Routing + Tool Execution

- route() chooses tool path only when intent is tool-mapped and slots are satisfied.
- Otherwise requests go to LLM reasoning or clarification path.
- Tool layer uses strict allowlist and per-tool argument validation.
- Implemented tools: get_weather, add_numbers, tell_joke, get_time.
- External weather data uses wttr.in with graceful network/error handling.

## Slide 12 - Stage 10 and 11: LLM Reasoning + Response Generation

- llm_client centralizes Gemini calls and error handling.
- ask_llm_reasoned handles open-ended queries with constrained instruction style.
- LLM also supports intent classification, query rewrite, and entity backfill.
- response.py standardizes final output formatting and cleanup.

## Slide 13 - Stage 12: Runtime Controls and Observability

- main.py provides interactive REPL orchestration.
- Runtime commands: intent-mode switching, debug mode, context inspection, exit.
- Debug mode reveals internals: intent, entities, route, missing slots, tokens.
- Design enables step-by-step explainability for demos and troubleshooting.

## Slide 14 - Validation and Test Coverage

- Automated sample validation via DAY2.test_samples.
- Coverage includes rule, ML, LLM, hybrid, extraction, context carry-over, safety, and reasoning.
- Multi-turn cases validate clarification and sticky context behavior.
- Edge-case checks include greetings, noise inputs, and ambiguous short utterances.

## Slide 15 - Outcomes, Limitations, and Next Improvements

- Outcomes: modular pipeline, robust fallback logic, safer input handling, explainable runtime behavior.
- Current limitations: probabilistic variance in LLM/semantic outcomes; occasional intent collisions on ambiguous phrases.
- Practical improvements: confidence-calibrated routing, richer NER, retrieval-augmented reasoning, persistent context store.
- Final takeaway: hybrid orchestration delivers strong reliability for structured tasks and flexibility for open-ended queries.
