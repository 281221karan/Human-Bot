from typing import TypedDict, Annotated
from database import conn 

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from functions_and_tools import calculator, search, augmentation_and_generation, retrieval, load_document_and_info


load_dotenv()

#---------------------------------------- ChatBotState -------------------------------------

class ChatBotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#--------------------------------------- LLM -----------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

tools = [calculator, search, augmentation_and_generation, retrieval, load_document_and_info]

llm_with_tools = llm.bind_tools(tools)

#--------------------------------------- NODES ---------------------------------------------

def chat_node(state: ChatBotState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

#--------------------------------------- Checkpointer ---------------------------------------

checkpointer = SqliteSaver(conn = conn)

#--------------------------------------- Graph ----------------------------------------------

graph = StateGraph(ChatBotState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

#--------------------------------------- Compile Graph ---------------------------------------

chatbot = graph.compile(checkpointer = checkpointer)


def retrieve_all_threads():
    all_threads = []
    for checkpoint in checkpointer.list(None):
        if checkpoint.config["configurable"]["thread_id"] not in all_threads:
            all_threads.append(checkpoint.config["configurable"]["thread_id"])
    return all_threads