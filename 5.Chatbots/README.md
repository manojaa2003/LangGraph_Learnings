# 🤖 LangGraph Chatbots — A Step-by-Step Learning Guide

This repository is a **hands-on study guide** for learning LangGraph. The four files are not just separate scripts — they are a **progressive learning path**, where each file builds on the previous one by introducing one new concept at a time.

> **Recommended approach:** Read and run the files in order — `1 → 2 → 3 → 4`. Each section below explains *what* is new, *why* it exists, and *how* it works in the code.

---

## 🗺️ Learning Path at a Glance

```
1.basic_chatbot.py
   └── Learn: StateGraph, nodes, edges, compiling a graph

2.chat_bot_with_tools.py
   └── Learn: Tools, bind_tools, ToolNode, conditional edges

3.in_memo_chatbot.py
   └── Learn: Checkpointers, MemorySaver, thread_id, multi-turn memory

4.persistant_memo_chatbot.py
   └── Learn: SqliteSaver, persistence across sessions
```

---

## ⚙️ Setup (Do This First)

### Requirements

- Python 3.9+
- A [Groq API key](https://console.groq.com/) — free to sign up
- A [Tavily API key](https://tavily.com/) — only needed for file 2

### Install Dependencies

```bash
pip install langchain langchain-groq langgraph langchain-community python-dotenv tavily-python
```

### Create a `.env` File

In the project folder, create a file named `.env` and add your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

`load_dotenv()` in each script automatically reads this file so your keys are never hardcoded.

---

## 📘 Lesson 1 — The Basic Chatbot

**File:** `1.basic_chatbot.py`

### What You Will Learn
- What a **StateGraph** is and why LangGraph uses one
- How to define **State** (the memory of your graph)
- How to create a **node** and connect it with **edges**
- How to **compile** and **run** the graph

### Core Concept: What is a Graph?

LangGraph models your application as a **graph** — a set of **nodes** (steps/functions) connected by **edges** (transitions). Think of it like a flowchart where data flows from one step to the next.

```
[START] ──► [chatbot node] ──► [END]
```

Every time the user sends a message, the graph runs this flow from start to finish.

### Core Concept: What is State?

**State** is the shared data that flows through the graph. Every node can read from it and write to it. You define it as a `TypedDict`:

```python
class Basic_stategraph(TypedDict):
    messages: Annotated[list, add_messages]
```

- `messages` is a list that holds the full conversation (user messages + AI replies)
- `Annotated[list, add_messages]` tells LangGraph to **append** new messages to the list instead of replacing it — this is the `add_messages` reducer

### Core Concept: What is a Node?

A node is just a **Python function** that takes the current state and returns an updated state:

```python
def chatbot(state: Basic_stategraph):
    bot_output = llm.invoke(state["messages"])
    return {"messages": [bot_output]}
```

The LLM receives the full message list, generates a reply, and the node returns it to be appended to state.

### Core Concept: Building and Compiling the Graph

```python
graph = StateGraph(Basic_stategraph)   # Create the graph with your state schema
graph.add_node("bot", chatbot)         # Register the node
graph.set_entry_point("bot")           # First node to run
graph.add_edge("bot", END)             # After "bot", the graph ends

app = graph.compile()                  # Compile into a runnable app
```

`compile()` converts the graph definition into an executable object. You must compile before you can invoke.

### Running the Graph

```python
response = app.invoke({
    "messages": [HumanMessage(content=user_input)]
})
```

Each call to `invoke` is **independent** — the bot has no memory of past turns. Every message you send is treated as a brand new conversation.

### ⚠️ Limitation of This Bot

Since there is no memory, the bot cannot answer follow-up questions. Try this:
```
user: My name is Manoj
AI: Nice to meet you, Manoj!

user: What is my name?
AI: I don't know your name.   ← it forgot already
```
This is the problem that files 3 and 4 solve.

**Run it:**
```bash
python 1.basic_chatbot.py
```

---

## 📗 Lesson 2 — Chatbot with Tools

**File:** `2.chat_bot_with_tools.py`

### What You Will Learn
- How to give an LLM access to **external tools**
- What `bind_tools` does
- How to use `ToolNode` to execute tool calls
- How to write a **conditional edge** to route between nodes

### Core Concept: Why Tools?

A plain LLM can only answer from what it was trained on. Tools let the LLM reach outside itself — searching the web, querying a database, calling an API. In this file, the tool is **Tavily**, a real-time web search engine.

### Core Concept: `bind_tools`

```python
search_tools = TavilySearchResults(max_results=2)
tools = [search_tools]
llm_with_tools = llm.bind_tools(tools=tools)
```

`bind_tools` tells the LLM *what tools are available* and *how to call them*. The LLM does not call the tool directly — it returns a **tool call request** (a structured message saying "please run this tool with these inputs"). LangGraph then handles the actual execution.

### Core Concept: ToolNode

`ToolNode` is a built-in LangGraph node that:
1. Reads the tool call request from the last AI message
2. Executes the tool
3. Returns the result as a `ToolMessage` appended to state

```python
tool_node = ToolNode(tools=tools)
```

You don't need to write any execution logic yourself — `ToolNode` handles it all.

### Core Concept: Conditional Edges

Not every user message needs a web search. The LLM decides. The `tool_router` function checks the last message and routes accordingly:

```python
def tool_router(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"   # LLM wants to use a tool
    else:
        return END           # LLM has a direct answer, stop here
```

`add_conditional_edges` uses this function to decide the next step at runtime:

```python
graph.add_conditional_edges("chatbot", tool_router)
```

### The Full Flow

```
User Input
    │
    ▼
[chatbot node]  ──── tool_router ────► [tool_node] ──► [chatbot node] ──► END
                                  ↘
                                   END  (no tool call needed)
```

When the LLM calls a tool, the graph loops back to `chatbot` so the LLM can read the tool result and form a final response.

### ⚠️ Note

This bot still has **no memory** between turns. Each invocation starts fresh. Memory is added in the next lesson.

**Run it:**
```bash
python 2.chat_bot_with_tools.py
```

---

## 📙 Lesson 3 — In-Memory Chatbot

**File:** `3.in_memo_chatbot.py`

### What You Will Learn
- What a **checkpointer** is and why it is needed
- How `MemorySaver` works
- What `thread_id` is and how it separates conversations
- Why the same graph can now handle multi-turn memory

### Core Concept: The Problem with `invoke`

In files 1 and 2, every call to `app.invoke({"messages": [HumanMessage(...)]})` **replaces** the state completely. The graph gets a fresh state with only the new message — it never sees previous messages.

### Core Concept: Checkpointers

A **checkpointer** is a storage layer that LangGraph uses to **save and restore state** between invocations. After each run, it saves a "checkpoint" (a snapshot of the state). On the next run, it loads the previous checkpoint and continues from there.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)
```

`MemorySaver` stores checkpoints **in RAM**. It is fast and requires no setup, but the memory disappears when your script stops.

### Core Concept: `thread_id`

Since one application can serve many different users, checkpointers use a `thread_id` to keep conversations separate:

```python
config = {
    "configurable": {
        "thread_id": 1
    }
}

response = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
```

Every call with the same `thread_id` is treated as **the same ongoing conversation**. Change the `thread_id` to start a completely fresh conversation without losing the old one.

### What Changed in the Invoke Call?

The `messages` you pass now only contain the **new** message from the user. LangGraph automatically loads the previous messages from the checkpoint and appends the new one. You never manually manage the history.

### Now the Bot Remembers

```
user: My name is Manoj
AI: Nice to meet you, Manoj!

user: What is my name?
AI: Your name is Manoj.   ← it remembers now ✅
```

### ⚠️ Limitation

Memory exists only **while the script is running**. Stop the script, restart it, and all history is gone. File 4 solves this.

**Run it:**
```bash
python 3.in_memo_chatbot.py
```

---

## 📕 Lesson 4 — Persistent Chatbot

**File:** `4.persistant_memo_chatbot.py`

### What You Will Learn
- The difference between in-memory and persistent storage
- How `SqliteSaver` works
- How to use an SQLite database as a checkpointer

### Core Concept: The Problem with MemorySaver

`MemorySaver` is great for development and testing, but it is **volatile** — it lives only in RAM. In a real application, users expect the chatbot to remember them even after the server restarts.

### Core Concept: SqliteSaver

`SqliteSaver` works exactly like `MemorySaver` from LangGraph's perspective — it is just a different backend that writes checkpoints to a **SQLite database file** on disk instead of RAM.

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("checkpointer.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

app = graph.compile(checkpointer=checkpointer)
```

- `checkpointer.sqlite` is created automatically in your working directory
- `check_same_thread=False` allows the connection to be used across different threads safely

### Everything Else Stays the Same

The graph definition, state, nodes, and edges are **identical** to file 3. The only change is swapping `MemorySaver` → `SqliteSaver`. This is the power of LangGraph's checkpointer abstraction — you can swap storage backends without changing any application logic.

### Persistent Memory in Action

```
--- First run ---
user: My name is Manoj
AI: Nice to meet you, Manoj!
user: exit

--- Restart the script (second run) ---
user: What is my name?
AI: Your name is Manoj.   ← remembered across restarts ✅
```

### The Upgrade Path

```
MemorySaver      → great for learning and prototyping
SqliteSaver      → great for local apps and single-user tools
PostgresSaver    → great for production, multi-user applications
```

**Run it:**
```bash
python 4.persistant_memo_chatbot.py
```

---

## 📌 Important Note on State Type Annotation

In files 3 and 4, the `messages` field in state is correctly annotated as a `list`:

```python
class Graph_state(TypedDict):
    messages: Annotated[list, add_messages]
```

This is important because the `add_messages` reducer expects a **list** of messages, not a string. Using `Annotated[str, add_messages]` is a common beginner mistake — it may not throw an error immediately but will cause confusing issues as your graph grows. Always use `list` here.

---

## 🔑 Key Terms Quick Reference

| Term | What It Means |
|------|--------------|
| **StateGraph** | The main LangGraph class — defines your graph with a state schema |
| **State** | A `TypedDict` that holds all data flowing through the graph |
| **Node** | A Python function that reads state and returns updated state |
| **Edge** | A connection between two nodes — defines the flow of execution |
| **Conditional Edge** | An edge where the next node is decided at runtime by a function |
| **`add_messages`** | A reducer that appends new messages to the list instead of replacing it |
| **`END`** | A special constant that terminates the graph execution |
| **`compile()`** | Converts the graph definition into a runnable application |
| **Checkpointer** | A storage layer that saves and restores graph state between runs |
| **`MemorySaver`** | An in-memory checkpointer — fast but not persistent |
| **`SqliteSaver`** | A file-based checkpointer — persists across restarts |
| **`thread_id`** | A unique ID that identifies a conversation thread |
| **`bind_tools`** | Registers tools with an LLM so it knows it can call them |
| **`ToolNode`** | A built-in node that executes tool calls made by the LLM |

---

## 🛠 Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — Graph-based agent framework
- **[LangChain](https://github.com/langchain-ai/langchain)** — LLM tooling and message types
- **[Groq](https://groq.com/)** — Fast LLM inference (`llama-3.1-8b-instant`)
- **[Tavily](https://tavily.com/)** — Real-time web search API (file 2 only)
- **SQLite** — Lightweight local database for persistent memory (file 4)

---

## 📚 What to Learn Next

Once you are comfortable with these four files, here are natural next steps:

- **Human-in-the-loop** — Pause graph execution to get approval from a human before continuing
- **Multi-agent graphs** — Multiple specialized agents collaborating inside one graph
- **Streaming** — Stream tokens to the user in real time instead of waiting for the full response
- **Custom tools** — Build your own tools (database queries, API calls, file readers) instead of using Tavily

---

*Built as part of learning Agentic AI development with LangChain and LangGraph.*
