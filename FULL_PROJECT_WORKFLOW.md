# SYNGENTA2 Full Workflow (Theory Only)

This document describes the complete workflow of the project as a technical lifecycle. It focuses only on theory, architecture, and decision flow. No installation or environment setup is included.

## 1. System Scope

The project has two implementation layers:

1. DAY1 layer: a monolithic exploration workflow in one file.
2. DAY2 layer: a modular, production-style pipeline split into dedicated stages.

The final architecture follows a hybrid NLP + LLM design where deterministic NLP handles structured tasks and LLM reasoning handles ambiguous or open-ended tasks.

## 2. Final Architecture Map (DAY2)

DAY2 is organized by stage responsibility:

1. safety.py: input validation and sanitization.
2. preprocessing.py: normalization, tokenization, stopword filtering.
3. semantic.py: short-query rewriting and semantic intent similarity matching.
4. intent.py: rule, ML, LLM, and hybrid intent detection.
5. entities.py: regex, dictionary, and LLM entity extraction.
6. context.py: short-term dialogue memory and slot carry-forward.
7. slots.py: required-slot validation and clarification prompting.
8. routing.py: decision between tool execution and LLM reasoning.
9. tools.py: controlled tool execution with whitelist and argument checks.
10. llm_client.py: shared LLM API access and reasoning entry points.
11. response.py: response post-processing.
12. pipeline.py: end-to-end orchestration.
13. main.py: interactive runtime loop and command controls.

## 3. Full Stage Order (Single User Turn)

The complete runtime order for each input is:

1. Input arrival.
2. Safety gate (validate + sanitize).
3. Preprocessing.
4. Semantic rewrite for short/underspecified queries.
5. Intent detection.
6. Semantic intent fallback if intent is unknown.
7. Entity extraction.
8. Context-based entity resolution.
9. Slot validation.
10. Routing decision.
11. Execution path:
   tool path or LLM path or clarification path.
12. Response generation and post-processing.
13. Context update for next turn.
14. Optional debug output (if debug mode is active).

## 4. Stage-by-Stage Theory With All Approaches

## Stage 1: Input Safety and Sanitization

Goal: block unsafe input and normalize raw text before NLP/LLM processing.

Approach families:

1. Rule-based pattern blocking:
   regex patterns for injection signatures, script tags, dangerous shell tokens.
2. Structural constraints:
   length caps, empty-input checks, basic character filtering.
3. Model-based safety classification:
   separate classifier or LLM moderation layer before task pipeline.
4. Hybrid safety chain:
   rules first for speed, classifier second for uncertain cases.

Project choice:

1. Rule-based pattern checks plus max-length and empty checks.
2. Sanitization via whitespace normalization and non-printable character stripping.

Why this matters:

1. Protects downstream stages from malicious or malformed inputs.
2. Reduces prompt-injection exposure at the entry point.

## Stage 2: Preprocessing

Goal: convert user text into stable, machine-friendly representations.

Approach families:

1. Text normalization:
   lowercasing, punctuation cleanup, whitespace normalization.
2. Tokenization:
   word tokenization for feature extraction and rule matching.
3. Stopword filtering:
   remove high-frequency low-signal words.
4. Morphological normalization:
   stemming or lemmatization for variant reduction.
5. Subword/tokenizer alignment:
   use model tokenizer directly for transformer-first systems.

Project choice:

1. Lowercase and punctuation normalization.
2. Tokenization using NLP toolkit methods.
3. Stopword-filtered token list for lightweight downstream use.

Why this matters:

1. Improves consistency in intent and entity processing.
2. Reduces sensitivity to punctuation and casing variation.

## Stage 3: Semantic Enrichment

Goal: improve understanding when user input is short, vague, or weakly matched.

Approach families:

1. Query rewriting:
   expand short user text to complete intent-bearing form.
2. Embedding-based similarity:
   compare user input against intent exemplars.
3. Lexical fallback similarity:
   bag-of-words cosine when embedding models are unavailable.
4. No-rewrite strict mode:
   skip rewrite for deterministic low-latency operation.

Project choice:

1. Rewrite only when input is very short.
2. Semantic intent matching as fallback after primary intent detection.
3. Embedding model preferred, lexical cosine fallback if unavailable.

Why this matters:

1. Raises recall for underspecified queries.
2. Avoids hard failure when direct keyword/rule evidence is missing.

## Stage 4: Intent Detection

Goal: map user text to one intent label that controls routing and slot schema.

Approach families:

1. Rule-based intent:
   deterministic keyword/regex patterns.
2. ML classifier intent:
   supervised model with confidence score.
3. LLM zero-shot intent:
   prompt model to output one label.
4. Hybrid chain:
   rule -> ML -> LLM fallback.
5. Ensemble voting:
   weighted combination of multiple detectors.

Project choice:

1. Supports four runtime modes: rule, ml, llm, hybrid.
2. Hybrid uses rule-first, then ML confidence checks, then LLM fallback.
3. Includes stricter handling for over-trigger-prone classes.

Why this matters:

1. Balances latency, determinism, and robustness.
2. Gives runtime control to trade speed vs flexibility.

## Stage 5: Entity Extraction

Goal: convert free text into structured arguments for tools.

Approach families:

1. Regex extraction:
   deterministic patterns for numbers, date/time, and simple location forms.
2. Dictionary/gazetteer matching:
   known entities matched by vocabulary.
3. LLM extraction:
   schema-targeted extraction when deterministic methods are insufficient.
4. CRF/sequence labeling:
   traditional statistical named entity recognition.
5. Transformer NER:
   contextual token labeling for higher linguistic coverage.
6. Hybrid merge strategy:
   cheap extractors first, LLM only for unresolved slots.

Project choice:

1. Regex extraction for numbers and location patterns.
2. Dictionary matching for known cities.
3. LLM extraction only when required slots remain unresolved.
4. Ordered merge: regex -> dictionary -> LLM.

Why this matters:

1. Controls cost and latency by deferring expensive extraction.
2. Preserves deterministic behavior whenever possible.

## Stage 6: Context Resolution

Goal: use prior dialogue state to fill missing values in current turn.

Approach families:

1. Last-turn memory:
   carry entities/intents from immediate previous turn.
2. Slot-state memory:
   track unresolved slots and fill them when user replies.
3. Intent-conditional carry-forward:
   reuse only if current and prior intents align.
4. Long-context summarization:
   periodically compress older dialogue into a state summary.
5. Retrieval memory:
   retrieve relevant history snippets by semantic similarity.

Project choice:

1. Maintains turn history with cap.
2. Tracks last intent, last entities, unresolved slots.
3. Resolves missing slots from context with intent-aware rules.

Why this matters:

1. Enables multi-turn completion workflows.
2. Reduces repeated questioning and improves conversational continuity.

## Stage 7: Slot Validation and Clarification

Goal: ensure required arguments exist before any tool is executed.

Approach families:

1. Static slot schema by intent:
   predefined required fields.
2. Dynamic schema from tool signatures:
   derive slot requirements directly from selected tool.
3. Clarification dialog policy:
   ask targeted follow-up for missing fields.
4. Confidence-aware slot repair:
   ask clarifying question when extraction confidence is low.

Project choice:

1. Static required-slot map by intent.
2. Validation checks for type and cardinality.
3. Clarification response path when slots are missing.

Why this matters:

1. Prevents invalid tool calls.
2. Converts extraction failures into controlled user prompts.

## Stage 8: Routing Decision

Goal: select the best execution path after intent and slot evaluation.

Approach families:

1. Deterministic intent router:
   known intent + complete slots -> tool.
2. Confidence-threshold router:
   route to LLM if confidence is below threshold.
3. Cost-aware router:
   prefer local tools to minimize API usage.
4. Policy-based router:
   include task type, safety class, and latency targets.
5. Learned router:
   train a model to choose best path from historical outcomes.

Project choice:

1. Deterministic route: tool only when intent has tool mapping and slots are satisfied.
2. Otherwise route to LLM or clarification path.

Why this matters:

1. Keeps deterministic tasks fast and predictable.
2. Preserves LLM for tasks requiring flexible reasoning.

## Stage 9: Tool Execution

Goal: run approved deterministic operations safely.

Approach families:

1. Whitelisted tool registry:
   execute only pre-approved tools.
2. Argument validator per tool:
   reject malformed or incomplete parameters.
3. Local deterministic tools:
   arithmetic, date/time, formatting.
4. External API tools:
   weather, search, retrieval, enterprise APIs.
5. Multi-tool chaining:
   route outputs of one tool into another.
6. Sandboxed tool runtime:
   process isolation and resource limits.

Project choice:

1. Allowed-tool whitelist.
2. Intent-to-tool mapping.
3. Per-tool argument validation before execution.
4. Graceful error returns for network/format failures.

Why this matters:

1. Prevents arbitrary code/tool invocation.
2. Produces stable structured outputs for final response composition.

## Stage 10: LLM Reasoning Path

Goal: answer open-ended or fallback queries through model reasoning.

Approach families:

1. Direct prompt answering:
   concise response from user query.
2. System-prompt constrained answering:
   enforce style and uncertainty disclosure.
3. Retrieval-augmented reasoning:
   attach retrieved documents to reduce hallucinations.
4. Tool-grounded reasoning:
   inject tool outputs into final generation prompt.
5. Chain-of-thought externalization control:
   keep internal reasoning private while returning concise answers.

Project choice:

1. Shared LLM client used for both extraction/classification prompts and reasoning.
2. Constrained reasoning prompt for uncertain/open-ended queries.
3. Tool fallback to LLM when deterministic path fails unexpectedly.

Why this matters:

1. Provides broad language capability beyond deterministic tool scope.
2. Maintains graceful degradation instead of hard failures.

## Stage 11: Response Generation

Goal: produce clean user-facing text regardless of source path.

Approach families:

1. Raw passthrough:
   return tool/LLM output directly.
2. Post-processing formatter:
   trim whitespace, collapse noisy line breaks.
3. Template-driven rendering:
   enforce standard output shape by route type.
4. Multi-part response packaging:
   include answer, confidence, source, and next-action hints.

Project choice:

1. Route-agnostic response generation entry point.
2. Post-processing cleanup for consistent formatting.

Why this matters:

1. Ensures output quality does not depend on source subsystem.
2. Makes logs and user interface behavior consistent.

## Stage 12: Context Persistence and Session Control

Goal: maintain dialogue continuity and runtime control over behavior.

Approach families:

1. In-memory per-session context.
2. Persistent store context across sessions.
3. Command-driven runtime controls.
4. Debug instrumentation mode.

Project choice:

1. In-memory context with bounded history.
2. Runtime commands to switch intent mode and inspect context.
3. Debug mode prints internal decision signals.

Why this matters:

1. Supports iterative testing of approaches in one session.
2. Improves explainability during development and demos.

## 5. End-to-End Decision Graph (Theory)

1. Receive user text.
2. If unsafe -> return blocked/error response.
3. Else sanitize and preprocess.
4. If query is very short -> attempt semantic rewrite.
5. Detect intent using configured mode.
6. If intent unknown -> semantic exemplar matching fallback.
7. Extract entities with ordered strategy (regex -> dictionary -> LLM).
8. Resolve missing values from dialogue context.
9. Validate required slots.
10. If slots missing -> return clarification response and wait for follow-up.
11. Else route:
    tool path if mapped and complete, otherwise LLM path.
12. Execute selected path.
13. Post-process response.
14. Update context state (intent, entities, unresolved slots).
15. Return final response.

## 6. Approach Matrix (Condensed)

| Stage | Approaches Covered | Typical Strength | Typical Risk | Project Usage |
|---|---|---|---|---|
| Safety | Rules, constraints, moderation, hybrid | Fast hard-blocking | False positives/negatives | Rule + constraints |
| Preprocessing | Normalization, tokenization, stopword, morphology | Stable downstream features | Over-normalization | Normalization + tokenization + stopword |
| Semantic | Rewrite, embeddings, lexical fallback | Better recall on short text | Rewrite drift | Rewrite + embedding fallback |
| Intent | Rule, ML, LLM, hybrid, ensemble | Flexible speed/quality control | Misclassification | Rule/ML/LLM/hybrid runtime modes |
| Entities | Regex, dictionary, LLM, NER, hybrid merge | Structured tool arguments | Extraction ambiguity | Regex + dictionary + LLM backfill |
| Context | Last-turn, slot-state, intent-aware carry | Multi-turn continuity | Wrong carry-over | Intent-aware slot/entity carry |
| Slots | Static schema, dynamic schema, clarification policy | Valid tool inputs | Over-questioning | Static schema + clarification |
| Routing | Deterministic, threshold, cost-aware, learned | Predictable control | Wrong path in edge cases | Deterministic tool-vs-LLM |
| Tools | Whitelist, validators, local/external APIs, chaining | Safe deterministic execution | API failures | Whitelist + validators + fallback |
| LLM | Direct QA, constrained prompt, RAG, tool-grounded | Broad capability | Hallucination | Constrained reasoning + fallback |
| Response | Passthrough, formatter, templates, structured payloads | Consistent UX | Information loss if over-trimmed | Formatter-based cleanup |

## 7. DAY1 to DAY2 Workflow Evolution

The project workflow evolves in this sequence:

1. Single-file prototype with integrated logic.
2. Introduction of multiple intent approaches.
3. Addition of hybrid fallback behavior.
4. Expansion to LLM-assisted extraction.
5. Tool-selection and execution control.
6. Migration to modular stage-by-stage architecture.
7. Addition of explicit safety gate.
8. Addition of semantic rewrite and similarity fallback.
9. Addition of dialogue context and slot carry-forward.
10. Finalized orchestrator with route-specific outcomes and response normalization.

## 8. Complete Lifecycle Blueprint (Generalized From This Project)

This project maps to a reusable lifecycle blueprint for hybrid NLP + LLM systems:

1. Intake and safety.
2. Linguistic preprocessing.
3. Semantic enrichment.
4. Intent inference.
5. Structured extraction.
6. Dialogue-state integration.
7. Slot completeness enforcement.
8. Policy routing.
9. Deterministic tool execution.
10. LLM reasoning fallback or augmentation.
11. Response normalization.
12. Session-state persistence.
13. Observability and debug controls.
14. Continuous evaluation and threshold tuning.

This ordering keeps deterministic components first, probabilistic components second, and expensive generative operations last. That is the core systems principle behind the implemented workflow.
