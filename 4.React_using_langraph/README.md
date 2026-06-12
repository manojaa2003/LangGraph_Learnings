# ReAct Agent using LangGraph

A from-scratch implementation of the classic **ReAct (Reason + Act)** pattern, built using LangGraph's `StateGraph` instead of the prebuilt `AgentExecutor`. The agent reasons about a query, decides which tool to call, executes it, observes the result, and repeats until it has enough information to answer.

---

## What It Does

Give it a question that requires real-time information, and it will:

- **Reason** about what it knows and what it still needs
- **Act** by calling a tool (web search or system datetime)
- **Observe** the tool's output
- **Repeat** this Reason → Act → Observe loop until it has a final answer
- Return the final answer once the LLM produces an `AgentFinish`

**Example query:**
```python
"how many days ago the SpaceX last rocket was launched?"
```

The agent will search the web for the latest SpaceX launch date, get the current date, calculate the difference, and return a final answer.

---

## How It Works

1. **`reasoning_node`** — invokes the ReAct agent runnable (Gemini 2.5 Flash + ReAct prompt from LangChain Hub). The LLM looks at the input and any prior tool outputs, then decides either to call a tool (`AgentAction`) or finish (`AgentFinish`).
2. **`act_node`** — looks up the chosen tool by name, invokes it with the LLM-provided input, and appends `(action, output)` to `intermediate_steps`.
3. **`should_continue`** — a conditional edge: if the outcome is `AgentFinish`, the graph ends; otherwise it loops back to `reasoning_node` with the new observation in context.

---

## Project Structure

```
6.React_using_langraph/
├── agent_state.py      # TypedDict defining the LangGraph state schema
├── react_runnable.py   # LLM, tools, and ReAct agent runnable setup
├── nodes.py             # reasoning_node and act_node implementations
└── react_graph.py       # StateGraph wiring, conditional edges, entry point
```

### File Descriptions

| File | Role |
|---|---|
| `agent_state.py` | Defines `AgentState`: `input` (user query), `agent_outcome` (`AgentAction` / `AgentFinish` / `None`), and `intermediate_steps` (accumulated tool call history via `operator.add`). |
| `react_runnable.py` | Sets up `ChatGoogleGenerativeAI` (Gemini 2.5 Flash), defines two tools (`TavilySearchResults` and a custom `get_system_datetime` tool), pulls the standard `hwchase17/react` prompt from LangChain Hub, and builds `agent_runnable` via `create_react_agent`. |
| `nodes.py` | `reasoning_node` invokes `agent_runnable` to get the next action or final answer. `act_node` matches the chosen tool by name and executes it, recording the result. |
| `react_graph.py` | Builds the `StateGraph`, sets `reasoning_node` as the entry point, wires the conditional `should_continue` edge (loop vs. `END`), compiles the graph, and runs a sample invocation. |

---

## How to Run

**Step 1 — Get your API keys**

| Key | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) — free tier available |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com/) — free tier available |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com/) — required to pull prompts from LangChain Hub and recommended for tracing |

**Step 2 — Navigate to the project**

```bash
cd 6.React_using_langraph
```

**Step 3 — Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**Step 4 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5 — Set up your API keys**

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT="Langraph_tuto"
```

**Step 6 — Run the agent**

```bash
python react_graph.py
```

This will run the ReAct loop on the default example query and print:
- The full final state (including `agent_outcome` and `intermediate_steps`)
- The final answer string extracted from `agent_outcome.return_values["output"]`

---

## Customisation

**Change the query** — in `react_graph.py`, update the `app.invoke(...)` call:

```python
result = app.invoke(
    {
        "input": "Your question here",
        "agent_outcome": None,
        "intermediate_steps": [],
    }
)
```

**Add more tools** — in `react_runnable.py`, define a new `@tool`-decorated function and add it to the `tools` list. `create_react_agent` and `act_node` will pick it up automatically by name.

**Swap the LLM** — replace `ChatGoogleGenerativeAI` with `ChatGroq` or any other LangChain chat model. Note: `create_react_agent` with `hub.pull("hwchase17/react")` works best with models that support text-based ReAct-style reasoning (not strict function-calling-only models).

---

## Key Dependencies & Versions

| Package | Version |
|---|---|
| `langchain` | 0.3.19 |
| `langchain-community` | 0.3.18 |
| `langchain-core` | 0.3.86 |
| `langchain-google-genai` | 2.0.9 |
| `langchain-groq` | 0.2.4 |
| `langchain-text-splitters` | 0.3.11 |
| `google-ai-generativelanguage` | 0.6.15 |
| `google-generativeai` | 0.8.6 |
| `groq` | 0.37.1 |
| `langgraph` | — |
| `pydantic` | 2.13.4 |
| `python-dotenv` | 1.2.2 |
| `typing_extensions` | 4.15.0 |

---

## Notes

- This implements ReAct **manually with `StateGraph`** rather than using LangGraph's prebuilt `create_react_agent` graph helper — useful for understanding exactly how the Reason → Act → Observe loop is constructed under the hood.
- `intermediate_steps` uses `Annotated[list, operator.add]` — this is LangGraph's reducer pattern, meaning each node's returned list is **appended** to the existing state rather than replacing it.
- The ReAct prompt is pulled live from **LangChain Hub** (`hwchase17/react`), so an internet connection (and optionally a LangSmith API key) is required at runtime.
- `from groq import Groq` is imported in `react_runnable.py` but unused — the active LLM is Gemini 2.5 Flash. Safe to remove or swap in if you want to use Groq instead.
- `act_node` calls `tool_function.invoke(**tool_input)` when `tool_input` is a dict — this assumes the dict keys match the tool function's parameter names exactly.
- **API note (as of mid-2026):** `create_react_agent` (from `langchain.agents`) and the `hub.pull("hwchase17/react")` prompt pattern used here are now deprecated in favor of LangChain's `create_agent` and binding tools directly to the model via `bind_tools` + `ToolNode`. The code in this project still runs correctly, but a future-proofed version would use these newer APIs.
