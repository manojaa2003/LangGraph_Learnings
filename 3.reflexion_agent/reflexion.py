from langgraph.graph import END,MessageGraph
from langchain_core.messages import BaseMessage,ToolMessage
from chains import first_responder_chain,second_responder_chain
from execute_tools import execute_tools
from typing import List


graph = MessageGraph()

Maxiterations = 3

graph.add_node("draft",first_responder_chain)
graph.add_node("revisor",second_responder_chain)
graph.add_node("execute_tools",execute_tools)

graph.set_entry_point("draft")

graph.add_edge("draft","execute_tools")
graph.add_edge("execute_tools","revisor")

def event_loop(state : List[BaseMessage]):
    count_tool_visits = sum(isinstance(item,ToolMessage) for item in state)
    num_iterations = count_tool_visits 
    if num_iterations > Maxiterations:
        return END
    return "execute_tools" 

graph.add_conditional_edges("revisor",event_loop)

app = graph.compile()

response = app.invoke("Write about how small businesses can leverage AI to grow")

print(response[-1].tool_calls[0]["args"]["answer"])
print(response)