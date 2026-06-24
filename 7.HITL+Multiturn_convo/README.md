# 🔄 Multi-Turn Conversations with Human-in-the-Loop (HITL) in LangGraph

---

## 📖 Overview

This project demonstrates a **stateful, multi-turn conversation workflow** where:

1. A user provides a LinkedIn topic
2. An LLM (via Groq's Llama 3.1) generates an initial LinkedIn post
3. A human reviews the generated content and provides feedback
4. The LLM refines the post based on feedback
5. Steps 3-4 repeat until the human is satisfied (types "done")

The entire workflow is managed by **LangGraph**, with state checkpointing via `MemorySaver` and explicit routing using `Command` objects. This is a production-ready pattern for any iterative content generation, code review, or multi-agent collaboration workflow.

---

## 🎯 Key Concepts

### 1. StateGraph — The Backbone

A `StateGraph` is LangGraph's way of defining a workflow as a state machine. Every node represents a step, and the state (stored in `graph_state`) flows through nodes, getting updated along the way.

```python
class graph_state(TypedDict):
    linkedin_topic : str
    generated_post : Annotated[List[str],add_messages]
    human_feedback : Annotated[List[str],add_messages]
```

**Why `Annotated[List[str], add_messages]`?**

The `add_messages` reducer means: *"When multiple messages come in, append them to a list instead of replacing."* This keeps a full conversation history, which the LLM uses for context in later refinements.

---

### 2. `interrupt()` — Pausing for Human Input

```python
user_feedback = interrupt(
    {
        "generated_post" : generated_post,
        "message" : "Provide feedback or type 'Done' to exit"
    }
)
```

When `interrupt()` is called:
- **Graph execution pauses**
- The interrupt message is returned to the caller
- The graph state is checkpointed (saved to `MemorySaver`)
- The caller can then collect human input and resume

This is **not** blocking. The thread doesn't wait — control returns to the caller. Execution resumes only when the caller invokes `Command(resume=...)` with new input.

---

### 3. `MemorySaver` — Stateful Checkpointing

```python
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

`MemorySaver` is an in-memory checkpointer that persists graph state across invocations. Each invocation is identified by a `thread_id`:

```python
thread_config = {
    "configurable" : {
        "thread_id" : uuid.uuid4() 
    }
}
```

This allows the same conversation to continue across multiple `app.stream()` and `app.invoke()` calls. The state is never lost.

---

### 4. `Command(goto=..., update=...)` — Explicit Routing

Instead of conditional edges, nodes can return a `Command` that says *exactly* where to go next:

```python
if user_feedback.lower() == "done":
    return Command(
        goto="end_node",
        update={"human_feedback" : ["Finalised"]}
    )
```

**Key advantage:** Routing logic lives inside the node that has the context to decide. No separate conditional function needed.

---

### 5. Multi-Turn Message History

The state uses `Annotated[List[str], add_messages]` to maintain a conversation history:

```python
human_feedback = state["human_feedback"] if "human_feedback" in state else ["no feedback yet"]
```

Over multiple turns, `human_feedback` accumulates:
- Turn 1: `["no feedback yet"]`
- Turn 2: `["Make it shorter"]`
- Turn 3: `["Make it shorter", "Add more emojis"]`
- Turn 4: `["Make it shorter", "Add more emojis", "Fix the tone"]`

The LLM sees this full history in the prompt, enabling context-aware refinements.

---

## 🏗️ Architecture

```
START
  ↓
┌─────────────────┐
│  model node     │
│ (Generate post) │
└────────┬────────┘
         ↓
┌─────────────────────────┐
│  human_node             │
│ (Collect feedback via   │
│  interrupt())           │
└────────┬────────────────┘
         │
    [user types feedback]
         │
    ┌────┴─────────┐
    │              │
[if "done"]   [if feedback]
    │              │
    ↓              ↓
┌──────────┐   model node
│ end_node │   (regenerate
│ (finish) │    with feedback)
└──────────┘    │
                └─→ human_node (loop)
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

## ⚙️ Setup & Installation

### 1. Clone / Navigate to Project

```bash
cd 5_multi_turn_convo
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com/

---

## ▶️ How to Run

```bash
python 5.multi_turn_convo.py
```

### Interactive Execution Flow

**Step 1:** The script prompts for a topic:
```
Enter the topic to create a linkedin content: AI agents in production
```

**Step 2:** The LLM generates an initial post and the workflow pauses:
```
[model_node] generated linkedIn post
Here's a LinkedIn post about AI agents in production...

[human_node] waiting for human feedback
```

**Step 3:** You provide feedback (or type "done"):
```
Type any feedback or type 'done' when finshed: Make it more concise
```

**Step 4:** The workflow resumes, the LLM regenerates the post with your feedback in mind, and asks again:
```
[model_node] generated linkedIn post
Here's a refined LinkedIn post...

Type any feedback or type 'done' when finshed: Add more statistics
```

**Step 5:** Repeat until satisfied. Type "done":
```
Type any feedback or type 'done' when finshed: done

final Human feedbeck: ['Make it more concise', 'Add more statistics']
final generated post: [AIMessage(content='Here\'s the final polished post...')]
```

---

## 🔍 Code Walkthrough

### Part 1: State Definition

```python
class graph_state(TypedDict):
    linkedin_topic : str                                      # The initial topic
    generated_post : Annotated[List[str],add_messages]        # Posts accumulate here
    human_feedback : Annotated[List[str],add_messages]        # Feedback history
```

Each field is typed. `Annotated[...]` with `add_messages` tells LangGraph: *"Append new messages, don't replace."*

---

### Part 2: The Model Node

```python
def model(state):
    post_topic = state["linkedin_topic"]
    human_feedback = state["human_feedback"] if "human_feedback" in state else ["no feedback yet"]

    prompt = f'''
    Linkedin post topic : {post_topic}
    Feedback : {human_feedback}

    generate a structured and well written LinkedIn post based on the given topic.
    consider the previous Human feedback to refine the content.
'''
    response = llm.invoke([
        SystemMessage(content="you are an expert LinkedIn content writer"),
        HumanMessage(content=prompt)
    ])

    generated_post = response.content

    return {
        "generated_post" : [AIMessage(content=generated_post)],
        "human_feedback" : human_feedback
    }
```

**What happens:**
1. Extract topic from state
2. Include all previous feedback in the prompt (context-aware refinement)
3. Call the LLM (Groq's Llama 3.1)
4. Return the generated post + existing feedback to state

The LLM sees the full feedback history, so each iteration is informed by all prior critiques.

---

### Part 3: The Human Node

```python
def human_node(state):
    generated_post = state["generated_post"]

    user_feedback = interrupt(
        {
            "generated_post" : generated_post,
            "message" : "Provide feedback or type 'Done' to exit"
        }
    )

    if user_feedback.lower() == "done":
        return Command(
            goto="end_node",
            update={"human_feedback" : ["Finalised"]}
        )
    return Command(
        goto="model",
        update={"human_feedback" : [user_feedback]}
    )
```

**What happens:**
1. Show the generated post to the human
2. Call `interrupt()` — execution pauses, state is checkpointed
3. When caller resumes with `Command(resume=...)`, the feedback becomes the return value of `interrupt()`
4. If "done" → route to end_node
5. Otherwise → route back to model with the feedback added to state

---

### Part 4: Graph Assembly

```python
graph = StateGraph(graph_state)

graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")
graph.set_finish_point("end_node")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

**Structure:**
- `START` → `model` (always start here)
- `model` → `human_node` (always)
- `human_node` → (dynamic, via `Command`)
  - `"done"` → `end_node` (finish)
  - feedback → `model` (loop)

---

### Part 5: Execution Loop

```python
for chunk in app.stream(input=initial_state, config=thread_config):
    for node_id, value in chunk.items():
        if (node_id == '__interrupt__'):
            while True:
                user_feedback = input("Type any feedback or type 'done' when finshed")
                app.invoke(Command(resume=user_feedback), config=thread_config)

                if user_feedback.lower() == 'done':
                    break
```

**What happens:**
1. `app.stream()` runs the graph until an `interrupt()` is encountered
2. When `__interrupt__` is yielded, the caller takes control
3. Collect user input via `input()`
4. Resume with `app.invoke(Command(resume=...))` — passes the feedback back to the paused `interrupt()`
5. Graph continues from where it left off
6. Repeat until user types "done"

---

## 💡 Key Design Patterns

### Pattern: Accumulating Feedback

Each turn appends to `human_feedback`, so the LLM always knows the full history:

```python
# Turn 1
human_feedback = ["Make it shorter"]

# Turn 2 — new feedback appended
human_feedback = ["Make it shorter", "Add examples"]

# Turn 3
human_feedback = ["Make it shorter", "Add examples", "Fix tone"]
```

The `add_messages` reducer handles this automatically — you return `[new_feedback]`, and it appends to the list.

---

### Pattern: Command-Based Routing

Instead of conditional edges, the node itself decides:

```python
if user_feedback.lower() == "done":
    return Command(goto="end_node", ...)
else:
    return Command(goto="model", ...)
```

This keeps all the logic in one place — cleaner than external conditional functions.

---

### Pattern: Stateful Resumption

The `thread_id` ensures the same conversation can resume across multiple invocations:

```python
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

# First invocation
for chunk in app.stream(input=initial_state, config=thread_config):
    # ... pauses at interrupt

# Second invocation (same thread_id)
app.invoke(Command(resume=user_feedback), config=thread_config)
# Continues from the exact checkpoint
```

---

## 🛠️ Customization Ideas

### 1. Different Content Types

Replace the LinkedIn post generator with any other LLM task:

```python
prompt = f'''
Code Review:
{code_snippet}

Feedback: {human_feedback}

Provide constructive feedback...
'''
```

### 2. Multiple LLMs

Use different models at different stages:

```python
# Fast generation
fast_llm = ChatGroq(model="llama-3.1-8b-instant")

# Detailed refinement
detailed_llm = ChatGroq(model="mixtral-8x7b-32768")
```

### 3. Persistent Checkpointing

Replace `MemorySaver` with a database:

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(connection_string="postgresql://...")
```

### 4. Auto-Feedback

Instead of human feedback, add a critic node that generates feedback automatically:

```python
def critic_node(state):
    post = state["generated_post"]
    feedback = llm.invoke(f"Review this post and suggest improvements: {post}")
    return Command(goto="model", update={"human_feedback": [feedback.content]})
```

---

## 📚 Related Concepts

**StateGraph** — Defines the workflow structure and message flow
**Annotated[..., add_messages]** — Automatic message history accumulation
**interrupt()** — Pause execution for human input
**MemorySaver** — Checkpointing and resumption across invocations
**Command(goto, update)** — Explicit routing with state updates
**Multi-turn conversations** — Using message history for context-aware refinement

---

## 🔗 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Checkpointing Guide](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#persistence)
- [LangGraph Message Types](https://langchain-ai.github.io/langgraph/concepts/low_level_concepts/#messages)
- [Groq API](https://console.groq.com/)
- [LangChain Academy](https://academy.langchain.com/)

---

## 📝 Common Issues & Troubleshooting

### Issue: "GROQ_API_KEY not found"

**Solution:** Ensure `.env` file exists and contains:
```
GROQ_API_KEY=your_actual_key
```

### Issue: "ThreadID mismatch" or state not resuming

**Solution:** Use the same `thread_config` across all `app.stream()` and `app.invoke()` calls.

### Issue: Feedback not being incorporated into next generation

**Solution:** Check that the prompt includes `Feedback : {human_feedback}`. The LLM needs to see the feedback in the prompt to act on it.

### Issue: Loop never exits

**Solution:** Make sure you're typing "done" (lowercase) or `user_feedback.lower() == "done"` will fail.

---

