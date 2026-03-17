# 🌾 Syngenta NLP + LLM Pipeline

An end-to-end conversational AI pipeline built for the **Syngenta Hackathon**.  
It combines rule-based NLP, ML classification, semantic similarity, and Gemini LLM to handle real-world multi-turn dialogues — with live weather, restaurant search, maths, jokes, and time tools.

---

## ✨ Features

| Capability | Description |
|---|---|
| 🌦️ Weather | Real-time weather via wttr.in / OpenWeatherMap |
| 🍽️ Restaurants | Nearby restaurant lookup via Geoapify |
| ➕ Maths | Add two numbers with full entity extraction |
| 😄 Jokes | Fresh, non-repeating jokes via Gemini |
| 🕐 Time | Current local time |
| 💬 General Q&A | Open-ended questions answered by Gemini |
| 🔒 Safety | SQL injection, XSS, prompt injection detection |
| 🧠 Reasoning trace | Every reply includes a short `Intent | Route | Tool` banner |
| 🔄 Multi-turn context | Sticky city carry-over, slot clarification, bare-city inference |

---

## 🏗️ Architecture

```
User Input
   │
   ▼  Step 12 ── Safety validation      (blocks injections / empty input)
   │
   ▼  Step 3  ── Preprocessing          (lowercase, tokenise, stopword removal)
   │
   ▼  Step 7  ── Semantic understanding (query rewriting for short inputs)
   │
   ▼  Step 4  ── Intent detection       (rule → ML → semantic → LLM hybrid)
   │
   ▼  Step 5  ── Entity extraction      (regex → dict → LLM)
   │
   ▼  Step 8  ── Dialogue context       (carry slots across turns)
   │
   ▼  Step 6  ── Slot checking          (ask for missing city / numbers)
   │
   ▼  Step 9  ── Routing                (tool vs. LLM)
   │
   ├──► Step 11 ── Tool execution       (weather / restaurants / math / joke / time)
   └──► Step 10 ── LLM reasoning        (Gemini for open-ended queries)
   │
   ▼  Step 13 ── Response generation    (reasoning banner + clean output)
```

### Intent Detection — 4-mode hybrid

| Mode | Strategy |
|---|---|
| `rule` | Keyword regex — fast, deterministic |
| `ml` | Naive Bayes classifier trained on labelled examples |
| `llm` | Gemini zero-shot classification |
| `hybrid` *(default)* | rule → ML → semantic similarity → LLM fallback |

---

## 📁 Project Structure

```
SYNGENTA2/
├── DAY1/                   # Day 1 exercises (functions, NLP basics)
│   ├── functions.py
│   └── NLP_LLM_BASICS.md
│
├── DAY2/                   # Full pipeline (main project)
│   ├── main.py             # Interactive REPL entry point
│   ├── pipeline.py         # Orchestrates all steps end-to-end
│   ├── intent.py           # Step 4  – Intent detection
│   ├── entities.py         # Step 5  – Entity extraction
│   ├── preprocessing.py    # Step 3  – Text normalisation & tokenisation
│   ├── semantic.py         # Step 7  – Embedding similarity & query rewriting
│   ├── slots.py            # Step 6  – Slot checking & clarification
│   ├── context.py          # Step 8  – Dialogue context management
│   ├── routing.py          # Step 9  – Tool vs. LLM routing
│   ├── tools.py            # Step 11 – Tool implementations
│   ├── llm_client.py       # Step 10 – Gemini API client
│   ├── response.py         # Step 13 – Response generation & reasoning banner
│   ├── safety.py           # Step 12 – Input validation & sanitisation
│   └── test_samples.py     # Automated regression test suite (36 cases)
│
├── .env                    # API keys (not committed)
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/kg290/SYNGENTA.git
cd SYNGENTA
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt   # or install manually — see below
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEOAPIFY_KEY=your_geoapify_key_here        # optional – for restaurant search
WEATHER_API_KEY=your_openweathermap_key    # optional – falls back to wttr.in
```

Get free keys:
- **Gemini** → [aistudio.google.com](https://aistudio.google.com)
- **Geoapify** → [geoapify.com](https://www.geoapify.com)
- **OpenWeatherMap** → [openweathermap.org](https://openweathermap.org/api) *(optional)*

### 3. Run the interactive chatbot

```bash
python -m DAY2.main
```

### 4. Run the regression test suite

```bash
python -m DAY2.test_samples
```

Expected output: **36/36 passed, 0 failed**

---

## 💬 Example Session

```
╔══════════════════════════════════════════════════╗
║        NLP + LLM Full Pipeline  –  DAY 2        ║
╚══════════════════════════════════════════════════╝

You: weather in pune
Intent detected: weather | Route: tool | Tool called: get_weather
Bot: Weather in Pune: Sunny. Temperature 32°C (feels like 30°C), humidity 15%, wind 4 km/h.

You: restaurant in nagpur
Intent detected: restaurant | Route: tool | Tool called: get_restaurants
Bot: Top restaurants in Nagpur: Barbeque Nation, Aryan Caterers, Ashoka, Pizza Hut, Guptaji Namkeen.

You: add 15 and 27
Intent detected: math | Route: tool | Tool called: add_numbers
Bot: 15 + 27 = 42

You: tell me a joke
Intent detected: joke | Route: tool | Tool called: tell_joke
Bot: Why did the bicycle fall over? Because it was two-tired!

You: restraurants in tokyo
Intent detected: restaurant | Route: tool | Tool called: get_restaurants
Bot: Top restaurants in Tokyo: たいめいけん, やぶ久, 日比谷 松本楼 ...

You: explain quantum entanglement
Intent detected: unknown | Route: llm
Bot: Quantum entanglement is a phenomenon where two or more particles ...

You: debug: weather
  [intent]   weather
  [entities] {}
  [route]    clarification
  [missing]  ['city']
Bot: Which city are you asking about?

You: context
Bot: Last intent: weather | Waiting for: ['city']

You: exit
Bot: Goodbye.
```

---

## 🛠️ REPL Commands

| Command | Description |
|---|---|
| `exit` / `quit` | End the session |
| `context` | Show current dialogue state |
| `intent-mode: rule\|ml\|llm\|hybrid` | Switch intent detection strategy |
| `debug: <text>` | Show full pipeline internals for any input |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `google-genai` | Gemini LLM API client |
| `nltk` | Tokenisation, stopword removal, Naive Bayes |
| `sentence-transformers` | Semantic embedding similarity |
| `requests` | Weather & restaurant API calls |
| `python-dotenv` | `.env` file loading |

---

## 🔒 Security

- **Input validation** blocks SQL injection, XSS, shell commands, and prompt injection attempts before any processing occurs.
- **Tool whitelist** — only `get_weather`, `get_restaurants`, `add_numbers`, `tell_joke`, `get_time` can be invoked.
- **API keys** are loaded from `.env` and are never committed to version control.

---

## 🧪 Test Suite

`DAY2/test_samples.py` runs **36 automated cases** covering:

- Rule-based, ML, hybrid, and semantic intent detection
- Entity extraction (regex numbers, dict-based city lookup)
- Multi-turn slot clarification flows
- Sticky-city context carry-over
- Safety rejection (SQL, XSS, prompt injection, empty input)
- Restaurant intent including typo inputs (`restraurants`)
