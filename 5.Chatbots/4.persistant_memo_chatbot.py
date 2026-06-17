from langgraph.graph import StateGraph,END,add_messages
from typing import TypedDict,Annotated
from langchain_core.messages import AIMessage,HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

conn = sqlite3.connect("checkpointer.sqlite",check_same_thread=False)

checkpointer = SqliteSaver(conn)

class Graph_state(TypedDict):
    messages : Annotated[list,add_messages]

def chatbot(state:Graph_state):
    bot_output = llm.invoke(state["messages"])
    return {
        "messages" : [bot_output]
    }

graph = StateGraph(Graph_state)

graph.add_node("chatbot",chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot",END)

app = graph.compile(checkpointer = checkpointer)

config = {"configurable" :{
    "thread_id" : 1
}
}

while True:
    user_input = input("user: ")
    if user_input in ["exit","clear"]:
        break
    else:
        response = app.invoke({
            "messages" : [HumanMessage(content=user_input)]
        },config=config)

        print(f"AI: {response['messages'][-1].content}")