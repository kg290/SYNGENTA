"""
DAY2 – Interactive REPL Entry Point.

Run from the project root:
    python -m DAY2.main

Commands recognised at runtime
-------------------------------
exit / quit             – close the session
context                 – print the current dialogue context summary
intent-mode: <mode>     – switch intent detection mode
                          valid modes: rule | ml | llm | hybrid
debug: <your text>      – process text and show all pipeline internals
<any other text>        – run through the full pipeline and get a response

Examples
--------
You: weather in Tokyo
Bot: Weather in Tokyo is sunny and 22°C.

You: add 15 and 27
Bot: 15 + 27 = 42

You: tell me a joke
Bot: <fresh LLM-generated joke>

You: debug: weather
  [intent]   weather
  [entities] {}
  [route]    clarification
  [missing]  ['city']
Bot: Which city are you asking about?

You: Paris
Bot: Weather in Paris is sunny and 22°C.
"""

from DAY2.context import DialogueContext
from DAY2.intent import INTENT_MODES
from DAY2.pipeline import run_pipeline

_BANNER = """\
╔══════════════════════════════════════════════════╗
║        NLP + LLM Full Pipeline  –  DAY 2        ║
╚══════════════════════════════════════════════════╝
Commands:
  intent-mode: rule | ml | llm | hybrid
  debug: <text>   → show pipeline internals
  context         → show current dialogue context
  exit / quit     → end session
──────────────────────────────────────────────────"""


def main() -> None:
    context = DialogueContext()
    intent_mode = "hybrid"

    print(_BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye.")
            break

        if not user_input:
            continue

        # ── Built-in commands ────────────────────────────────────────────────

        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Goodbye.")
            break

        if user_input.lower() == "context":
            print(f"Bot: {context.get_summary()}")
            continue

        if user_input.lower().startswith("intent-mode:"):
            requested = user_input.split(":", 1)[1].strip().lower()
            if requested in INTENT_MODES:
                intent_mode = requested
                print(f"Bot: Intent mode set to '{intent_mode}'.")
            else:
                valid = ", ".join(sorted(INTENT_MODES))
                print(f"Bot: Unknown mode. Choose from: {valid}")
            continue

        # ── Debug mode ───────────────────────────────────────────────────────

        debug_mode = False
        if user_input.lower().startswith("debug:"):
            user_input = user_input.split(":", 1)[1].strip()
            debug_mode = True

        # ── Full pipeline run ────────────────────────────────────────────────

        result = run_pipeline(user_input, context, intent_mode=intent_mode)

        if debug_mode:
            print(f"  [intent]   {result.intent}")
            print(f"  [entities] {result.entities}")
            print(f"  [route]    {result.route_taken}")
            print(f"  [missing]  {result.missing_slots}")
            tokens = result.preprocessed.get("tokens", [])
            print(f"  [tokens]   {tokens}")

        print(f"Bot: {result.response}")


if __name__ == "__main__":
    main()
