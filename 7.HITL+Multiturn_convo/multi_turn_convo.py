from dotenv import load_dotenv
from langgraph.graph import StateGraph,END,add_messages,START
from langgraph.types import Command,interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict,Annotated,List
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchain_groq import ChatGroq
import uuid

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

class graph_state(TypedDict):
    linkedin_topic : str
    generated_post : Annotated[List[str],add_messages]
    human_feedback : Annotated[List[str],add_messages]

def model(state):
    print("model generating content")

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

    print(f"[model_node] generated linkedIn post\n {generated_post}")

    return {
        "generated_post" : [AIMessage(content=generated_post)],
        "human_feedback" : human_feedback
    }

def human_node(state):

    print("\n[human_node] waiting for human feedback")

    generated_post = state["generated_post"]

    user_feedback = interrupt(
        {
            "generated_post" : generated_post,
            "message" : "Provide feedback or type 'Done' to exit"
        }
    )

    if user_feedback.lower() == "done":
        return Command(
            goto= "end_node",
            update= {
               "human_feedback" : ["Finalised"]
            }
        )
    return Command(
        goto="model",
        update={
                "human_feedback" : [user_feedback],
         }
    )

def end_node(state):
    print(f'''\n final Human feedbeck: {state["human_feedback"]}''')
    print(f'''\n final generated post: {state["generated_post"]}''')
    return {
        "human_feedback" : state["human_feedback"],
        "generated_post" : state["generated_post"],
    }

graph = StateGraph(graph_state)

graph.add_node("model",model)
graph.add_node("human_node",human_node)
graph.add_node("end_node",end_node)

graph.add_edge(START,"model")
graph.add_edge("model","human_node")
graph.set_finish_point("end_node")

checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

thread_config = {
    "configurable" : {
        "thread_id" : uuid.uuid4() 
    }
}

linked_topic = input("Enter the topic to create a linkedin content") 

initial_state = {
    "linkedin_topic" : linked_topic,
    "generated_post" : [],
    "human_feedback" : []

}

for chunk in app.stream(input=initial_state,config=thread_config):
    for node_id,value in chunk.items():
        if (node_id == '__interrupt__'):
            while True:
                user_feedback = input("Type any feedback or type 'done' when finshed")
                app.invoke(Command(
                    resume=user_feedback),
                    config = thread_config
                )

                if user_feedback.lower() == 'done':
                    break

        
    