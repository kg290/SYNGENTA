"""
Step 8 – Dialogue Context.

Maintains a lightweight conversation state across turns so that
follow-up answers can be matched to earlier questions.

Example interaction this enables:
    User   : weather
    System : Which city are you asking about?
    User   : Tokyo
    System : Weather in Tokyo is sunny.  ← city carried from context

The context is kept in-memory (per session).  No persistence to disk.
History is capped at the last 10 turns to avoid unbounded growth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_MAX_HISTORY = 10


@dataclass
class Turn:
    """A single conversation turn."""
    user_input: str
    intent: str
    entities: dict
    response: str


@dataclass
class DialogueContext:
    """
    Step 8 – Dialogue Context.

    Attributes:
        history         – ordered list of recent Turn objects
        last_intent     – intent label from the most recent turn
        last_entities   – entities extracted in the most recent turn
        unresolved_slots – slot names that were missing in the last turn
    """

    history: list[Turn] = field(default_factory=list)
    last_intent: str = ""
    last_entities: dict = field(default_factory=dict)
    unresolved_slots: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(
        self,
        user_input: str,
        intent: str,
        entities: dict,
        response: str,
        unresolved: list[str] | None = None,
    ) -> None:
        """
        Record a completed turn and refresh the context state.

        Args:
            user_input:  Raw text the user typed.
            intent:      Detected intent for this turn.
            entities:    Entities extracted for this turn.
            response:    The response sent back to the user.
            unresolved:  Slot names that were still missing (caused a clarification).
        """
        self.history.append(Turn(user_input, intent, entities, response))
        self.last_intent = intent
        self.last_entities = dict(entities)
        self.unresolved_slots = unresolved or []

        # Trim to keep only the most recent _MAX_HISTORY turns
        if len(self.history) > _MAX_HISTORY:
            self.history = self.history[-_MAX_HISTORY:]

    # ------------------------------------------------------------------
    # Slot resolution from context
    # ------------------------------------------------------------------

    def resolve_from_context(
        self,
        current_entities: dict,
        current_intent: str,
    ) -> dict:
        """
        Attempt to fill missing slots using information from prior turns.

        Rules applied (in order):
        1. If the previous turn left unresolved slots AND the current
           intent matches the previous intent, carry over any values
           from the previous entities that are still missing now.
        2. Always carry the city across consecutive weather or restaurant
           queries so "weather?" after "weather in Tokyo" still works.
        3. If the current input contains no recognised entities but we
           are awaiting a city answer, treat the entire (cleaned) input
           as a possible city reply.

        Args:
            current_entities: Entities already extracted from the current turn.
            current_intent:   Intent detected for the current turn.

        Returns:
            Merged entity dict.
        """
        merged = dict(current_entities)

        # Rule 1 – carry unresolved slots from previous turn
        if self.unresolved_slots and self.last_intent == current_intent:
            for slot in self.unresolved_slots:
                if slot not in merged and slot in self.last_entities:
                    merged[slot] = self.last_entities[slot]

        # Rule 2 – sticky city for repeated weather/restaurant intent
        if (
            current_intent in ("weather", "restaurant")
            and current_intent == self.last_intent
            and "city" not in merged
            and "city" in self.last_entities
        ):
            merged["city"] = self.last_entities["city"]

        return merged

    # ------------------------------------------------------------------
    # Diagnostic helper
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Return a one-line summary of the current context state."""
        if not self.history:
            return "(no prior context)"
        last = self.history[-1]
        parts = [f"Last intent: {last.intent}"]
        if last.entities:
            parts.append(f"Entities: {last.entities}")
        if self.unresolved_slots:
            parts.append(f"Waiting for: {self.unresolved_slots}")
        return " | ".join(parts)
