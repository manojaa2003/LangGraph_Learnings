from langchain_google_genai import ChatGoogleGenerativeAI
from groq import Groq
from langchain.agents import tool,create_react_agent
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
from langchain import hub
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",verbose=True)

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_datetime(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current system date and time.
    """
    curent_datetime = datetime.datetime.now()
    formatted_datetime = curent_datetime.strftime(format)
    return formatted_datetime

tools = [search_tool,get_system_datetime]

react_prompt = hub.pull("hwchase17/react")

agent_runnable = create_react_agent(llm=llm, tools=tools,prompt=react_prompt)