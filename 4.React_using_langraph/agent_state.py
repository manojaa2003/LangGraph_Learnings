from langchain_core.agents import AgentAction,AgentFinish
from typing import List,Union,TypedDict,Annotated
import operator

class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction,AgentFinish,None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

