from dotenv import load_dotenv
from langchain_core.agents import AgentAction,AgentFinish
from langgraph.graph import StateGraph,END

from nodes import reasoning_node,act_node
from agent_state import AgentState  

load_dotenv()

graph = StateGraph(AgentState)

graph.set_entry_point("reasoning_node")
graph.add_node("reasoning_node", reasoning_node)
graph.add_node("act_node", act_node)

def should_continue(state:AgentState):
    if isinstance(state["agent_outcome"],AgentFinish):
        return END
    else:
        return "act_node"

graph.add_conditional_edges("reasoning_node", should_continue)
graph.add_edge("act_node", "reasoning_node")

app = graph.compile()

result = app.invoke(
    {
        "input":"how many days ago the SpaceX last rocket was launched?",
        "agent_output" : None,
        "intermediate_steps": [],
    }
)

print(result)

print(result["agent_outcome"].return_values["output"],"final_output")



