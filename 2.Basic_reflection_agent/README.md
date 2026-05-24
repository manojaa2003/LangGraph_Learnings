# Basic Reflection Agent

A LangGraph-based AI agent that uses a **generate → reflect → revise** loop to iteratively improve Twitter posts. A single LLM (Gemini 2.5 Flash) plays two roles: a tweet writer and a viral-content critic — each refining the output across multiple passes.

---

## How It Works

```
User Request → Generate Tweet → Reflect & Critique → Revise Tweet → (repeat) → Final Tweet
```

1. **Generate** — Gemini 2.5 Flash acts as a *Twitter techie influencer assistant* and writes the best possible tweet for the user's request. On subsequent rounds, it incorporates the critique from the previous step.
2. **Reflect** — Gemini 2.5 Flash switches role to a *viral Twitter influencer critic* and grades the tweet, providing detailed recommendations on length, style, virality, and tone.
3. **Loop** — The reflection output is fed back as a `HumanMessage` to the generator. The loop runs until the message history exceeds **4 messages** (i.e., 2 full generate–reflect cycles), then returns the final tweet.

---

## Project Structure

```
2.Basic_reflection_agent/
├── chains.py      # Prompt templates, LLM setup, and generation/reflection chains
└── basic.py       # LangGraph graph definition and main entry point
```

### File Descriptions

| File | Purpose |
|---|---|
| `chains.py` | Defines `generation_prompt` and `reflection_prompt` using `ChatPromptTemplate`. Initialises `ChatGoogleGenerativeAI` with `gemini-2.5-flash` and exports `generation_chain` and `reflection_chain`. |
| `basic.py` | Builds a `MessageGraph` with two nodes (`generate`, `reflect`), a conditional edge that ends the loop after 4 messages, and a fixed edge from `reflect` back to `generate`. Runs a sample invocation and prints the graph in ASCII and Mermaid format. |

---

## How to Run

Follow these steps in order to get the agent running.

**Step 1 — Get a Google AI API key**

Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and create a free API key for Gemini.

**Step 2 — Clone / download the project**

```bash
cd 2.Basic_reflection_agent
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

**Step 5 — Set up your API key**

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

This is loaded automatically by `python-dotenv` — no other configuration needed.

**Step 6 — Run the agent**

```bash
python basic.py
```

You will see:
- The graph structure printed in **Mermaid** and **ASCII** formats
- The generate → reflect loop running for the default example topic
- The full message history printed on completion

---

## Customisation

**Change the input topic** — in `basic.py`, update the `app.invoke(...)` call:

```python
response = app.invoke(HumanMessage(content="Your topic here"))
```

**Change the number of iterations** — the loop stops when `len(state) > 4` (2 full cycles). To add more refinement passes:

```python
def should_continue(state):
    if len(state) > 6:   # 3 full generate-reflect cycles
        return END
    return REFLECT
```

---

## Architecture Diagram

```
┌──────────┐       ┌──────────┐
│ generate │──────▶│ reflect  │
│ (writer) │◀──────│ (critic) │
└────┬─────┘       └──────────┘
     │
     │  len(state) > 4
     ▼
    END
```

---

## Key Dependencies & Versions

| Package | Version |
|---|---|
| `langchain` | 0.3.19 |
| `langchain-community` | 0.3.18 |
| `langchain-core` | 0.3.86 |
| `langchain-google-genai` | 2.0.9 |
| `langchain-text-splitters` | 0.3.11 |
| `google-ai-generativelanguage` | 0.6.15 |
| `google-generativeai` | 0.8.6 |
| `groq` | 0.37.1 |
| `langchain-groq` | 0.2.4 |
| `pydantic` | 2.13.4 |
| `python-dotenv` | 1.2.2 |
| `typing_extensions` | 4.15.0 |

---

## Notes

- Unlike the Reflexion Agent, this project does **not** use web search — all refinement is done purely through LLM self-critique.
- Both chains use the **same LLM** (`gemini-2.5-flash`) but with different system prompts, making it a lightweight dual-role setup.
- The reflection output is wrapped in a `HumanMessage` before being fed back into the graph. This is intentional — it simulates user feedback, keeping the conversation structure valid for the generator prompt.
- `groq` and `langchain-groq` are listed in requirements for consistency with the broader project environment but are not used in this agent.
