from langchain_groq import ChatGroq
from langgraph import graph
from langgraph.graph import StateGraph,END,add_messages
from typing import TypedDict,Annotated
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant")

class Basic_stategraph(TypedDict):
    messages : Annotated[list,add_messages]

def chatbot(state :Basic_stategraph):
    bot_output = llm.invoke(state["messages"])
    return {
        "messages" : [bot_output]
    }

graph = StateGraph(Basic_stategraph)

graph.add_node("bot",chatbot)
graph.set_entry_point("bot")
graph.add_edge("bot",END)

app = graph.compile()

while True:
    user_input = input("user: ")
    if user_input in ["exit","end"]:
        break
    else:
        response = app.invoke({
            "messages":  [HumanMessage(content=user_input)]
        })
    
    print(f"AI: {response['messages'][-1].content}")