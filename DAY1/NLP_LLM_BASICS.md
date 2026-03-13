# NLP + LLM Basics (Beginner Guide)

This guide is a very simple map for learning how NLP and LLMs work together.

## What You Built

You now have one final file: functions.py

That file includes:
- Text preprocessing (tokenization + stopword removal)
- Rule-based intent detection
- Number extraction
- Tool functions (weather, math, joke)
- Gemini LLM integration
- LLM-based city extraction
- Hybrid NLP + LLM routing
- LLM tool selection and execution
- Final agent loop

## Why Combine NLP + LLM

NLP handles fast, predictable tasks:
- Simple intent rules
- Structured extraction
- Basic preprocessing

LLM handles flexible tasks:
- Understanding open-ended questions
- Extracting entities from messy text
- Choosing tools when rules are not enough

Together they form a practical hybrid system.

## Learning Flow (Step by Step)

1. Preprocess text
2. Detect intent with rules
3. Extract entities (numbers/city)
4. Call tools
5. Add LLM fallback
6. Use LLM for extraction
7. Use LLM for tool selection
8. Build final agent loop

## How to Run

From project folder:

.\venv\Scripts\python.exe .\functions.py

## Try These Inputs

1. preprocess: I am learning how to integrate NLP with LLM systems
2. add 5 and 10
3. tell me a joke
4. weather in Tokyo
5. explain black holes in one line
6. exit

## What To Observe

1. For math and joke, rules should be enough and very fast.
2. For weather, city extraction can use LLM.
3. For unknown requests, LLM gives a direct response.
4. In agent mode, LLM can choose a tool and then respond naturally.

## Very Basic Mental Model

User input -> NLP preprocessing -> intent detection -> extraction -> tool call -> LLM response

When rules fail:
User input -> LLM reasoning -> tool choice -> tool output -> final response

## Beginner Practice Tasks

1. Add a new tool: get_date()
2. Add a new intent: greeting
3. Let LLM choose from 5 tools instead of 3
4. Log each decision to understand routing
5. Compare rule-only vs LLM-only vs hybrid

## Common Beginner Mistakes

1. Running multiple files with duplicated logic
2. Not loading .env before calling Gemini
3. Expecting perfect JSON from LLM every time
4. No fallback when extraction fails
5. Mixing demo code with production flow

## Next Learning Goal

Try adding memory (store previous user messages) so the agent can answer with context.
