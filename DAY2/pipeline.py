"""
Final Pipeline – Orchestrates all steps end-to-end.

Full flow
---------
User input
    │
    ▼  Step 12 ── Safety validation (reject or sanitize)
    │
    ▼  Step 3  ── Preprocessing (normalise, tokenise)
    │
    ▼  Step 7  ── Semantic understanding
    │              • Rewrite very short queries via LLM
    │
    ▼  Step 4  ── Intent detection (rule / ML / LLM / hybrid)
    │              • Semantic fallback if still unknown
    │              • Context inference if slots were unresolved
    │
    ▼  Step 5  ── Entity extraction (regex → dict → LLM)
    │
    ▼  Step 8  ── Dialogue context resolution
    │              • Carry slots from previous turns
    │
    ▼  Step 6  ── Slot checking
    │              • Missing slots → ask user for clarification
    │
    ▼  Step 9  ── Routing decision (tool vs. LLM)
    │
    ├──► Step 11 ── Tool execution (get_weather / add_numbers / …)
    │
    └──► Step 10 ── LLM reasoning (ask_llm_reasoned)
    │
    ▼  Step 13 ── Response generation (post-process and return)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from DAY2.context import DialogueContext
from DAY2.entities import extract_entities
from DAY2.intent import detect_intent
from DAY2.llm_client import ask_llm_reasoned
from DAY2.preprocessing import preprocess
from DAY2.response import generate_response
from DAY2.routing import route
from DAY2.safety import sanitize_input, validate_input
from DAY2.semantic import rewrite_query, semantic_intent_match
from DAY2.slots import check_slots, clarification_prompt
from DAY2.tools import execute_tool, get_tool_name_for_intent


@dataclass
class PipelineResult:
    """
    Carries the outcome of a single pipeline run.

    Fields
    ------
    response       – final text to show the user
    intent         – detected intent label
    entities       – extracted (and context-resolved) entities
    route_taken    – 'tool', 'llm', 'llm_fallback', or 'clarification'
    missing_slots  – slot names that were absent (caused clarification)
    preprocessed   – output dict from the preprocessing step
    """

    response: str
    intent: str = "unknown"
    entities: dict = field(default_factory=dict)
    route_taken: str = ""
    missing_slots: list[str] = field(default_factory=list)
    preprocessed: dict = field(default_factory=dict)


def run_pipeline(
    user_input: str,
    context: DialogueContext,
    intent_mode: str = "hybrid",
) -> PipelineResult:
    """
    Run the full NLP + LLM pipeline for a single user turn.

    Args:
        user_input:  Raw text typed by the user.
        context:     Mutable DialogueContext that persists across turns.
        intent_mode: 'rule', 'ml', 'llm', or 'hybrid' (default).

    Returns:
        PipelineResult with the response and all intermediate state.
    """

    # ── Step 12: Safety ──────────────────────────────────────────────────────
    safe, reason = validate_input(user_input)
    if not safe:
        response = generate_response(
            "error",
            f"I cannot process that request: {reason}",
            reasoning={"intent": "unknown", "route_taken": "blocked"},
        )
        context.update(user_input, "unknown", {}, response)
        return PipelineResult(response=response, intent="unknown", route_taken="blocked")

    clean_input = sanitize_input(user_input)

    # ── Step 3: Preprocessing ────────────────────────────────────────────────
    preprocessed = preprocess(clean_input)
    normalized = preprocessed["normalized"] or clean_input.lower()

    # ── Greeting / social shortcut ────────────────────────────────────────────
    # Respond immediately without running any NLP steps.
    from DAY2.semantic import is_social, is_identity_question
    if is_social(normalized):
        _greet_map = {
            "bye": "Goodbye!", "goodbye": "Goodbye!", "cya": "Goodbye!",
            "see you": "Goodbye!", "later": "Goodbye!", "farewell": "Goodbye!",
            "thanks": "You're welcome!", "thank you": "You're welcome!",
            "thx": "You're welcome!", "ty": "You're welcome!", "cheers": "You're welcome!",
            "thankyou": "You're welcome!",
        }
        raw_response = _greet_map.get(
            normalized,
            "Hello! Ask me about the weather, restaurants, maths, a joke, or the current time."
        )
        response = generate_response(
            "shortcut",
            raw_response,
            reasoning={"intent": "unknown", "route_taken": "shortcut"},
        )
        context.update(user_input, "unknown", {}, response)
        return PipelineResult(
            response=response,
            intent="unknown",
            route_taken="shortcut",
            preprocessed=preprocessed,
        )

    # ── Identity question shortcut ──────────────────────────────────────────
    # Describe the bot’s own capabilities without delegating to Gemini.
    if is_identity_question(normalized):
        raw_response = (
            "I am an NLP + LLM assistant. Here is what I can do:\n"
            "  \u2022 Weather     \u2013 e.g. 'weather in Tokyo'\n"
            "  \u2022 Restaurants \u2013 e.g. 'find restaurants in Paris'\n"
            "  \u2022 Maths       \u2013 e.g. 'add 5 and 10'\n"
            "  \u2022 Jokes       \u2013 e.g. 'tell me a joke'\n"
            "  \u2022 Time        \u2013 e.g. 'what time is it'\n"
            "  \u2022 General questions \u2013 I will reason and answer."
        )
        response = generate_response(
            "shortcut",
            raw_response,
            reasoning={"intent": "unknown", "route_taken": "shortcut"},
        )
        context.update(user_input, "unknown", {}, response)
        return PipelineResult(
            response=response,
            intent="unknown",
            route_taken="shortcut",
            preprocessed=preprocessed,
        )

    # ── Bare number guard ─────────────────────────────────────────────────────
    # A standalone number has no operator or second operand, so it cannot
    # be resolved as a maths request. Asking for clarification is more
    # helpful than trying to split its digits into two numbers.
    import re as _re_inline
    if _re_inline.match(r"^-?\d+(\.\d+)?$", normalized.strip()):
        response = generate_response(
            "shortcut",
            "I need more context. For maths try: 'add 5 and 10' or 'sum 3 and 7'.",
            reasoning={"intent": "unknown", "route_taken": "shortcut"},
        )
        context.update(user_input, "unknown", {}, response)
        return PipelineResult(
            response=response,
            intent="unknown",
            route_taken="shortcut",
            preprocessed=preprocessed,
        )

    # ── Step 7: Semantic understanding – query rewriting ─────────────────────
    # Expand ambiguous short queries before intent detection.
    working_text = normalized
    if len(preprocessed["tokens"]) <= 3:
        expanded = rewrite_query(clean_input)
        if expanded and expanded.lower() != clean_input.lower():
            working_text = expanded.lower()

    # ── Step 4: Intent detection ──────────────────────────────────────────────
    intent = detect_intent(working_text, mode=intent_mode)

    # Semantic fallback: if rule/ML/LLM all return unknown, try embedding match
    if intent == "unknown":
        semantic_hit = semantic_intent_match(working_text)
        if semantic_hit:
            intent = semantic_hit

    # Context inference: if intent is still unknown but we have pending
    # unresolved slots, assume the user is answering the previous clarification.
    if intent == "unknown" and context.unresolved_slots:
        intent = context.last_intent

    # ── Step 5: Entity extraction ──────────────────────────────────────────────
    entities = extract_entities(clean_input, intent)

    # ── Step 8: Dialogue context resolution ───────────────────────────────────
    entities = context.resolve_from_context(entities, intent)

    # ── Context city inference ─────────────────────────────────────────────────
    # If the user types a bare city name (intent=unknown) after a weather or
    # restaurant turn, treat it as a new request of the same type for that city.
    if intent == "unknown" and context.last_intent in ("weather", "restaurant") and "city" in entities:
        intent = context.last_intent

    # ── Step 6: Slot checking ──────────────────────────────────────────────────
    slots_ok, missing = check_slots(intent, entities)

    # ── Step 9: Routing ────────────────────────────────────────────────────────
    decision = route(intent, slots_ok)

    # ── Step 11 / 10 / clarification: Execute ──────────────────────────────────
    tool_called = ""

    if not slots_ok and missing:
        # Ask the user for the missing information
        raw_response = clarification_prompt(missing)
        route_taken = "clarification"

    elif decision == "tool":
        tool_called = get_tool_name_for_intent(intent) or ""
        tool_result, tool_err = execute_tool(intent, entities)
        if tool_result:
            raw_response = tool_result
            route_taken = "tool"
        else:
            # Tool failed despite routing – fall back to LLM with error context
            raw_response = ask_llm_reasoned(user_input)
            route_taken = "llm_fallback"

    else:
        # LLM handles all open-ended or unknown requests
        raw_response = ask_llm_reasoned(user_input)
        route_taken = "llm"

    # ── Step 13: Response generation ───────────────────────────────────────────
    response = generate_response(
        route_taken,
        raw_response,
        reasoning={
            "intent": intent,
            "route_taken": route_taken,
            "tool_called": tool_called,
            "missing_slots": missing,
        },
    )

    # Persist this turn into the dialogue context
    context.update(
        user_input=user_input,
        intent=intent,
        entities=entities,
        response=response,
        unresolved=missing if not slots_ok else [],
    )

    return PipelineResult(
        response=response,
        intent=intent,
        entities=entities,
        route_taken=route_taken,
        missing_slots=missing,
        preprocessed=preprocessed,
    )
