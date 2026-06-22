from langgraph.graph import StateGraph,END,add_messages
from typing import TypedDict,Annotated
from langgraph.types import Command

class Graph_state(TypedDict):
    text: str

def node_a(state):
    print("Node A")
    return Command(
        goto="node_b",
        update={
            "text" : state["text"] + "a"
        }
    )

def node_b(state):
    print("Node B")
    return Command(
        goto="node_c",
        update={
            "text" : state["text"] + "b"
        }
    )

def node_c(state):
    print("Node C")
    return Command(
        goto=END,
        update={
            "text" : state["text"] + "c"
        }
    )
graph = StateGraph(Graph_state)

graph.add_node("node_a",node_a)
graph.add_node("node_b",node_b)
graph.add_node("node_c",node_c)

graph.set_entry_point("node_a")
app = graph.compile()

cur_state ={
    "text" : ""
}
reponse = app.invoke(cur_state)
print(reponse)