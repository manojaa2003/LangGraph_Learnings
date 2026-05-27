# Reflexion Agent — AI Research Assistant

A LangGraph-powered AI research agent that **drafts, self-critiques, searches the web, and revises** its answers iteratively — delivering a well-cited, verified response to any research question.

Unlike a simple Q&A chatbot, this agent behaves like a researcher: it identifies gaps in its own knowledge, searches for evidence to fill them, and rewrites its answer with numbered citations and references — all automatically.

---

## What It Does

Give it any research question and it will:

- Write an initial ~80 word expert answer
- Critique its own response (what's missing, what's unnecessary)
- Generate 2–3 targeted search queries based on that critique
- Search the web in real time via Tavily
- Revise the answer with citations and a references section
- Repeat this cycle up to **3 times** for deeper accuracy

**Example input:**
```
"How can small businesses leverage AI to grow?"
```

**What you get back:**
- A concise, revised answer (~80 words)
- Inline numerical citations `[1]`, `[2]`
- A references section with verified source links

---

## How It Works

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  DRAFT                                               │
│  Llama 3.3-70B answers the question (~80 words)      │
│  + self-critique (missing / superfluous)             │
│  + generates 2–3 search queries                      │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  EXECUTE TOOLS                                       │
│  Tavily runs each search query → returns web results │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  REVISE                                              │
│  LLM rewrites answer using search results            │
│  + adds [1][2] citations + References section        │
│  + addresses critique from previous round            │
└──────────────────┬───────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  Iterations < 3?    │
        │  YES → back to      │
        │        Execute Tools│
        │  NO  → END          │
        └─────────────────────┘
```

---

## Project Structure

```
Reflexion_Agent/
├── schema.py           # Pydantic v2 models for structured LLM output
├── chains.py           # Prompts + LangChain chains (draft & revise)
├── execute_tools.py    # Tavily web search execution
├── reflexion.py        # LangGraph graph — entry point
├── requirements.txt    # All dependencies with pinned versions
└── .env                # API keys (never commit this to Git)
```

### File Descriptions

| File | Role |
|---|---|
| `schema.py` | Defines `Reflection` (critique), `Answer_question` (draft output), and `Revised_answer` (revised output with references) as Pydantic v2 models for structured tool calling. |
| `chains.py` | Shared `actor_prompt_template` with two specialisations: `first_responder_chain` (draft) and `second_responder_chain` (revise). Both use Llama 3.3-70B via Groq with tool-calling bound to the Pydantic schemas. |
| `execute_tools.py` | Reads the last AI message's tool calls, extracts search queries, runs each through `TavilySearchResults`, and returns `ToolMessage` objects with JSON results. |
| `reflexion.py` | Wires the `MessageGraph`: `draft → execute_tools → revisor`, with a conditional edge counting `ToolMessage` iterations to decide loop or end. |

---

## How to Run

**Step 1 — Get your API keys**

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/) — free tier available |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com/) — free tier available |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) — free tier available |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com/) — free tier available (used for tracing & observability) |

**Step 2 — Clone / download the project**

```bash
cd Reflexion_Agent
```

**Step 3 — Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**Step 4 — Install dependencies**

A `requirements.txt` is already included in the project. Just run:

```bash
pip install -r requirements.txt
```

**Step 5 — Set up your API keys**

Create a `.env` file in the project root:

```env
# LLM Inference
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Web Search
TAVILY_API_KEY=your_tavily_api_key_here

# LangSmith Tracing (optional but recommended)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT="Langraph_tuto"
```

Loaded automatically by `python-dotenv` — no other config needed.

> **Note on LangSmith:** The `LANGSMITH_*` variables enable tracing and observability via [LangSmith](https://smith.langchain.com/). Every run will be logged to your `Langraph_tuto` project where you can visualise the full graph execution, inspect inputs/outputs at each node, and debug issues. Set `LANGSMITH_TRACING=false` to disable it.

**Step 6 — Run the agent**

```bash
python reflexion.py
```

The agent will run the full draft → search → revise loop and print the final cited answer to the terminal.

**Step 7 — Test the draft chain only (optional)**

```bash
python chains.py
```

Runs only `first_responder_chain` (no search, no revision) — useful for testing prompts or debugging structured output.

---

## Customisation

**Change the research question** — in `reflexion.py`, update:

```python
response = app.invoke("Your research question here")
```

**Change the number of research iterations** — default is 3. Increase for deeper research:

```python
Maxiterations = 5   # more web search + revision cycles
```

**Change the number of search results per query** — in `execute_tools.py`:

```python
tavily_tool = TavilySearchResults(max_results=5)   # default is 3
```

**Switch to Google Gemini** — `ChatGoogleGenerativeAI` is already imported in `chains.py`. To use it, add `GOOGLE_API_KEY` to your `.env` and swap the `llm` variable:

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
```

---

## Architecture Diagram

```
┌────────────┐     ┌───────────────┐     ┌──────────────┐
│   draft    │────▶│ execute_tools │────▶│   revisor    │
│            │     │               │     │              │
│ • answer   │     │ • Tavily web  │     │ • rewrite    │
│ • critique │     │   search      │     │ • citations  │
│ • queries  │     │ • 3 results   │     │ • references │
└────────────┘     └───────────────┘     └──────┬───────┘
                                                 │
                          ┌──────────────────────┤
                          │                      │
                   iterations < 3          iterations ≥ 3
                          │                      │
                          ▼                      ▼
                   execute_tools               END
```

---

## Key Dependencies & Versions

| Package | Version |
|---|---|
| `langchain` | 0.3.19 |
| `langchain-community` | 0.3.18 |
| `langchain-core` | 0.3.86 |
| `langchain-groq` | 0.2.4 |
| `langchain-google-genai` | 2.0.9 |
| `langchain-text-splitters` | 0.3.11 |
| `google-ai-generativelanguage` | 0.6.15 |
| `google-generativeai` | 0.8.6 |
| `groq` | 0.37.1 |
| `pydantic` | 2.13.4 |
| `python-dotenv` | 1.2.2 |
| `typing_extensions` | 4.15.0 |

---

## Notes

- The agent counts `ToolMessage` objects in state to track iterations — each web search round adds one, making it a reliable loop counter.
- Revised answers are capped at ~80 words (excluding the references section) — this is enforced via the revision prompt, not code.
- `MessageGraph` is used instead of `StateGraph` because the full state is a flat list of `BaseMessage` objects — simpler and sufficient for this architecture.
- Tavily's free tier supports up to 1,000 searches/month — well within limits for development and testing.
