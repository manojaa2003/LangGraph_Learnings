from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph,END,add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict,Annotated
from dotenv import load_dotenv
load_dotenv()

llm  = ChatGroq(model="llama-3.1-8b-instant")

search_tools = TavilySearchResults(max_results=2)

tools = [search_tools]

llm_with_tools = llm.bind_tools(tools=tools)

class BasicState(TypedDict):
    messages : Annotated[list,add_messages]

def chatbot(state:BasicState):
    bot_output = llm_with_tools.invoke(state["messages"])
    return {
        "messages" : [bot_output]
    }

def tool_router(state):
    last_message = state["messages"][-1]
    if (hasattr(last_message,"tool_calls") and len(last_message.tool_calls)>0):
        return "tool_node"
    else:
        return END

tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicState)

graph.add_node("chatbot",chatbot)
graph.add_node("tool_node",tool_node)

graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot",tool_router)
graph.add_edge("tool_node","chatbot")

app = graph.compile()

while True:
    user_input = input("user: ")
    if user_input in ["exit","end"]:
        break
    else:
        response = app.invoke({
            "messages":  [HumanMessage(content=user_input)]
        })
    
    print(response)
        
