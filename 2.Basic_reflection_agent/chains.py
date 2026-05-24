from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            "generate the best possible twitter post possible for the user request."
            "if the user provides a critique, respond with a revised version of your previous attempts"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a viral twitter influencer grading a tweet, generate critique and recommendations for the user tweet."
            "always provide a detailed recommendation, including requests for length, virality, style etc,"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm