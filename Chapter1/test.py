import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS

# 1. Load environment variables from .env file
load_dotenv()

# 2. Initialize the LLM
# ChatOpenAI will automatically look for os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Define the Search Tool using the direct DDGS library
@tool
def search_tool(query: str) -> str:
    """Search the web for current facts like inflation rates."""
    with DDGS() as ddgs:
        # We join the snippets found into one string for the AI to read
        results = [r['body'] for r in ddgs.text(query, max_results=3)]
        return "\n".join(results)

# 4. Define the Calculator Tool
@tool
def python_calculator(code: str) -> str:
    """Useful for mathematical calculations. Input: '1.05 * 0.07'"""
    try:
        # Clear globals/locals for a tiny bit more safety in eval
        return str(eval(code, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

tools = [search_tool, python_calculator]

# 5. Create the Agent State Machine (The LangGraph way)
# This replaces the old AgentExecutor/AgentType
agent_executor = create_react_agent(llm, tools)

# 6. Execute the task
def main():
    user_query = "Search the current inflation rate in India and calculate 5% of it"
    
    print(f"--- Processing: {user_query} ---")
    
    # LangGraph uses a dictionary with a list of messages
    inputs = {"messages": [HumanMessage(content=user_query)]}
    
    try:
        result = agent_executor.invoke(inputs)
        # The final answer is the last AI message in the list
        print("\nFINAL ANSWER FROM AGENT:")
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    main()