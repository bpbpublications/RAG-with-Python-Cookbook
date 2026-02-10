import os
import re
from dotenv import load_dotenv # This will work now!
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from duckduckgo_search import DDGS

# 1. Load Environment
load_dotenv()

# 2. Tools logic
@tool
def search_tool(query: str) -> str:
    """Search for current inflation data in India."""
    with DDGS() as ddgs:
        results = [r['body'] for r in ddgs.text(query, max_results=3)]
        return "\n".join(results)

@tool
def calculator(expression: str) -> str:
    """Calculate math. Input: '1.33 * 0.05'"""
    try:
        # Clean the input to ensure it's just math
        clean_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        return str(eval(clean_expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

tools = [search_tool, calculator]

# 3. Model Initialization (Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0
).bind_tools(tools)

# 4. Agent Logic
def assistant(state: MessagesState):
    sys_msg = SystemMessage(content="Search for the inflation rate first, then calculate 5% of it.")
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

# 5. Build Graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()

# 6. Run
if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="India's current inflation and 5% of that?")]}
    for chunk in graph.stream(inputs, stream_mode="values"):
        print(f"\n--- {chunk['messages'][-1].type.upper()} ---")
        print(chunk['messages'][-1].content)