from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from schema import Answer_question,Revised_answer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
import time,datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

pydantic_parser = PydanticToolsParser(tools=[Answer_question,Revised_answer])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         you are an expert AI researcher.
         current_time : {time}
         1.{first_instruction}
         2.Reflect and critique your answer. Be severe to maximize the improvement.
         3.After the reflection, list 2-3 search queries separately for researching improvements. do not include them in reflection.
        """
         ),
         MessagesPlaceholder(variable_name="messages"),
         ("human","Answer the user question above using required format"),
    ]
).partial(
    time = lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "provide a detailed ~80 word answer"
)

revise_istructions = """
Revise your previous answer using the new information.
->you should use previuos critique to add important information to your answer.
->you must include numerical citations in your revised answer to ensure it can be verified.
-> Add refernces section on bottom of your answer (which does not count towards the 80 word limit) with links to the sources you used for revision in the form of:
References:
[1] <link>
[2] <link>

-> you should use previous critique to remove superfluous information from your answer and make sure it is not more than 80 words.
"""

second_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = revise_istructions
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools = [Answer_question],tool_choice="Answer_question") 
second_responder_chain = second_responder_prompt_template | llm.bind_tools(tools = [Revised_answer],tool_choice="Revised_answer") 



response = first_responder_chain.invoke([HumanMessage(content="write me a blog post on how small businesses can leverage AI to grow")])

print(response)

