
import json
from langchain_core.messages import AIMessage,ToolMessage, HumanMessage,BaseMessage
from langchain_community.tools import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=3)

def execute_tools(state : list[BaseMessage]):
    last_ai_message : AIMessage =  state[-1]

    if not hasattr(last_ai_message,"tool_calls") or not last_ai_message.tool_calls:
        return []

    tool_meaasges = []
    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["Answer_question","Revised_answer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries",[])

            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result

            tool_meaasges.append(
                ToolMessage(
                    content = json.dumps(query_results),
                    tool_call_id = call_id,
                    name=tool_call["name"]
                )
            )
    return tool_meaasges

