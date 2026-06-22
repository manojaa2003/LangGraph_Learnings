# 🤖 Human-in-the-Loop (HITL) with LangGraph

---

## 📖 Overview

This module explores **Human-in-the-Loop (HITL)** design patterns in LangGraph — a critical concept in production-grade agentic systems where human oversight, approval, or feedback is injected into an otherwise automated AI workflow.

Rather than letting an agent run end-to-end without checks, HITL patterns allow you to **pause execution**, collect human input, and **resume or redirect** the graph — giving you fine-grained control over AI behaviour.

---

## 🗂️ Module Contents

| File | Pattern | Description |
|------|---------|-------------|
| `1.Human_in_the_loop.py` | **Feedback Loop** | LLM generates a LinkedIn post; human reviews and optionally provides feedback to regenerate |
| `2.human_loop_command_class.py` | **Command Routing** | Demonstrates `Command` objects for explicit node-to-node routing and state updates |
| `3.hu_lp_using_interrupt.ipynb` | **`interrupt()` + Resume** | Graph pauses mid-execution at `node_b` for human branching input (C or D) using `MemorySaver` |
| `4.hu_lp_using_approve.ipynb` | **Tool Approval Gate** | LLM + Tavily search agent; human must approve before any tool call executes via `interrupt_before` |

---

## 🧠 Concepts Covered

### 1. Manual HITL via `input()` — `1.Human_in_the_loop.py`

The earliest and simplest HITL pattern: use Python's built-in `input()` inside a conditional edge function to ask the human whether to approve or revise.

```
generate_post → [human reviews] → post_in_linkedin
                              ↘ collect_feedback → generate_post (loop)
```

**Key idea:** The conditional edge `get_review_decision` reads the last LLM output, prints it, and branches based on human input — no checkpointers required.

---

### 2. `Command` Objects for Explicit Routing — `2.human_loop_command_class.py`

Instead of relying on `add_conditional_edges`, nodes return a `Command` object that explicitly declares:
- **`goto`** — which node to route to next
- **`update`** — what state changes to apply

```python
return Command(
    goto="node_b",
    update={"text": state["text"] + "a"}
)
```

**Key idea:** Removes the need for a separate routing function. The node itself decides where to go.

---

### 3. `interrupt()` + `MemorySaver` + `Command(resume=...)` — `3.hu_lp_using_interrupt.ipynb`

The modern LangGraph HITL primitive. When `interrupt()` is called inside a node, the graph **suspends execution** and saves state to a checkpointer. Execution resumes only when the caller passes `Command(resume=<value>)`.

```python
# Inside node_b — graph pauses here
human_input = interrupt("do you want to go to C or D, type C/D")

# Later, from the caller
app.invoke(Command(resume="D"), config=config)
```

**Requires:** `MemorySaver` (or any checkpointer) + a `thread_id` in config.

**Key idea:** The graph is truly stateful across invocations. The checkpoint stores everything; the `resume` value is injected as the return value of `interrupt()`.

---

### 4. `interrupt_before` — Tool Approval Gate — `4.hu_lp_using_approve.ipynb`

A production-ready pattern where a human must approve before any tool call is executed. The graph is compiled with `interrupt_before=["tools"]`, which automatically pauses before the `tools` node runs.

```python
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])
```

Flow:
```
chatbot → [PAUSE: human approves] → tools → chatbot → END
```

If the question doesn't need a tool (e.g., "What is the capital of India?"), the graph completes without pausing. If it does (e.g., "What is the weather in Delhi?"), it halts at the tool boundary.

To approve and continue:
```python
app.stream(None, config=config, stream_mode="values")
```

**Key idea:** No changes to node logic are needed — the interrupt is declarative at compile time.

---

## 🔑 Key API Reference

| API | Purpose |
|-----|---------|
| `interrupt(value)` | Pause graph execution; return `value` to human |
| `Command(goto=..., update=...)` | Explicit node routing with state update |
| `Command(resume=...)` | Resume a paused graph with a human-provided value |
| `MemorySaver()` | In-memory checkpointer for persisting graph state |
| `graph.compile(interrupt_before=[...])` | Declarative pause before specified nodes |
| `app.get_state(config)` | Inspect current state + `.next` to see pending node |

---

## 🏗️ Graph Architecture Summary

```
File 1 — Feedback Loop
──────────────────────
HumanMessage → generate_post → get_review_decision ──► post_in_linkedin → END
                      ▲              │
                      │    (feedback)▼
                      └── collect_feedback

File 3 — interrupt() Branch
────────────────────────────
node_a → node_b ──[interrupt]──► (human: C or D)
                        │
              ┌─────────┴──────────┐
           node_c               node_d
              └─────────┬──────────┘
                        END

File 4 — Tool Approval
───────────────────────
chatbot ──► [interrupt_before: tools] ──► (human approves) ──► tools ──► chatbot ──► END
```

---

## ⚙️ Environment Setup

### 1. Clone / navigate to this module

```bash
cd 8.Human_in_the_loop
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here    # Required for file 4 only
```

---

## 📦 Dependencies

| Package | Version |
|---------|---------|
| `langchain` | 0.3.19 |
| `langchain-community` | 0.3.18 |
| `langchain-core` | 0.3.86 |
| `langchain-google-genai` | 2.0.9 |
| `langchain-groq` | 0.2.4 |
| `langchain-text-splitters` | 0.3.11 |
| `google-ai-generativelanguage` | 0.6.15 |
| `google-generativeai` | 0.8.6 |
| `groq` | 0.37.1 |
| `langgraph` | latest |
| `pydantic` | 2.13.4 |
| `python-dotenv` | 1.2.2 |
| `typing_extensions` | 4.15.0 |

---

## ▶️ Running the Examples

### Python scripts (Files 1 & 2)

```bash
python 1.Human_in_the_loop.py
python 2.human_loop_command_class.py
```

### Jupyter Notebooks (Files 3 & 4)

```bash
jupyter notebook
```

Open `3.hu_lp_using_interrupt.ipynb` or `4.hu_lp_using_approve.ipynb` and run cells sequentially. Pay attention to the multi-step execution — some cells intentionally pause the graph and others resume it.

---

## 💡 When to Use Each Pattern

| Scenario | Recommended Pattern |
|----------|-------------------|
| Simple scripts / CLIs | `input()` inside conditional edge (File 1) |
| Clean routing logic inside nodes | `Command(goto=..., update=...)` (File 2) |
| Pause & resume across multiple invocations | `interrupt()` + `MemorySaver` (File 3) |
| Gating tool/API calls before execution | `interrupt_before=[...]` at compile time (File 4) |
| Production agents needing audit trails | `interrupt_before` + persistent checkpointer (DB) |

---

## 📚 Learning Series Index

> Modules completed so far in this Agentic AI learning path:

- Module 1–7 — LangChain fundamentals, chains, agents, tools, memory, RAG
- **Module 8 — Human-in-the-Loop** ← *You are here*
- Module 9+ — Multi-agent systems, LangGraph advanced patterns *(coming soon)*

---

## 🔗 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph HITL Guide](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
- [Groq API](https://console.groq.com/)
- [Tavily Search API](https://tavily.com/)
- [LangChain Academy](https://academy.langchain.com/)

---

*Built as part of a structured self-learning path in Agentic AI — LangChain & LangGraph Bootcamp.*
