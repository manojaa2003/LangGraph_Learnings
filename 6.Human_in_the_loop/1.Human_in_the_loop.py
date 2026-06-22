from langgraph.graph import StateGraph,END,add_messages
from typing import TypedDict,Annotated
from langchain_core.messages import AIMessage,HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

class Graph_State(TypedDict):
    messages : Annotated[str,add_messages]

def generate_post(state):
    bot_output = llm.invoke(state["messages"])
    return {
        "messages" : [bot_output]
    }

def get_review_decision(state):
    last_output = state["messages"][-1].content
    print(last_output)
    print("\n")
    decision = input("any feedback? (yes/no): ")
    if(decision.lower() == "yes"):
        return "collect_feedback"
    else:
        return "post_in_linkedin"
    
def post_in_linkedin(state):
    final_post = state["messages"][-1].content
    print(final_post)
    print("\n")
    print("posted sucessfully✅")

def collect_feedback(state):
    feedback = input("how can i improve this post ")
    return{
            "messages" : [HumanMessage(content=feedback)]
    }

graph = StateGraph(Graph_State)

graph.add_node("generate_post",generate_post)
graph.add_node("post_in_linkedin",post_in_linkedin)
graph.add_node("collect_feedback",collect_feedback)

graph.set_entry_point("generate_post")
graph.add_conditional_edges("generate_post",get_review_decision)
graph.add_edge("collect_feedback","generate_post")
graph.add_edge("post_in_linkedin",END)

app = graph.compile()

response = app.invoke({
    "messages" : [HumanMessage(content="how the AI agents are benifical in research and development?")]
})

print(response)
