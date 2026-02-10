# pip install langgraph langchain langchain-huggingface


import pandas as pd
from typing import TypedDict, Optional

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, END

# --------------------------------------------------
# LLM Setup
# --------------------------------------------------
text2text_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_length=512
)

llm = HuggingFacePipeline(pipeline=text2text_pipeline)

# --------------------------------------------------
# Shared State
# --------------------------------------------------
class SalesAgentState(TypedDict):
    file_path: str
    df: Optional[pd.DataFrame]
    revenue_summary: Optional[str]
    trend_summary: Optional[str]
    draft_report: Optional[str]
    feedback: Optional[str]
    approved: bool
    iteration: int

# --------------------------------------------------
# Agents
# --------------------------------------------------

def data_loader_agent(state: SalesAgentState) -> SalesAgentState:
    df = pd.read_excel(state["file_path"])
    print("✅ Data loaded")
    return {**state, "df": df}


def revenue_agent(state: SalesAgentState) -> SalesAgentState:
    total = state["df"]["revenue"].sum()
    summary = f"Total Revenue: ₹{total:,.2f}"
    return {**state, "revenue_summary": summary}


def trend_agent(state: SalesAgentState) -> SalesAgentState:
    region_sales = state["df"].groupby("region")["revenue"].sum()
    top_region = region_sales.idxmax()
    return {**state, "trend_summary": f"Top Performing Region: {top_region}"}


def report_generator_agent(state: SalesAgentState) -> SalesAgentState:
    prompt = f"""
Generate a professional sales report.

Insights:
{state['revenue_summary']}
{state['trend_summary']}

Previous feedback (if any):
{state.get('feedback', 'None')}

Improve clarity, structure, and executive tone.
"""
    report = llm.invoke(prompt)
    print(f"📝 Report generated (iteration {state['iteration']})")
    return {**state, "draft_report": report}


def critic_agent(state: SalesAgentState) -> SalesAgentState:
    critique_prompt = f"""
You are a strict business report reviewer.

Evaluate the report below for:
- Clarity
- Professional tone
- Actionable insights

Report:
{state['draft_report']}

Respond with:
APPROVED or REJECTED
Then provide brief feedback.
"""
    critique = llm.invoke(critique_prompt)

    approved = "APPROVED" in critique.upper()
    print("🔍 Critic decision:", "APPROVED" if approved else "REJECTED")

    return {
        **state,
        "approved": approved,
        "feedback": critique,
    }


def decision_agent(state: SalesAgentState):
    if state["approved"] or state["iteration"] >= 3:
        print("🏁 Workflow complete")
        return END
    else:
        print("🔁 Refining report...")
        return "generate_report"

# --------------------------------------------------
# LangGraph
# --------------------------------------------------
workflow = StateGraph(SalesAgentState)

workflow.add_node("load_data", data_loader_agent)
workflow.add_node("calculate_revenue", revenue_agent)
workflow.add_node("analyze_trends", trend_agent)
workflow.add_node("generate_report", report_generator_agent)
workflow.add_node("critic", critic_agent)

workflow.set_entry_point("load_data")

workflow.add_edge("load_data", "calculate_revenue")
workflow.add_edge("calculate_revenue", "analyze_trends")
workflow.add_edge("analyze_trends", "generate_report")
workflow.add_edge("generate_report", "critic")

workflow.add_conditional_edges(
    "critic",
    decision_agent,
    {
        "generate_report": "generate_report",
        END: END
    }
)

sales_graph = workflow.compile()

# --------------------------------------------------
# Runner
# --------------------------------------------------
def run_autonomous_sales_agent(file_path: str):
    state: SalesAgentState = {
        "file_path": file_path,
        "df": None,
        "revenue_summary": None,
        "trend_summary": None,
        "draft_report": None,
        "feedback": None,
        "approved": False,
        "iteration": 1,
    }

    while True:
        result = sales_graph.invoke(state)
        state["iteration"] += 1
        if result["approved"] or state["iteration"] > 3:
            break

    print("\n📊 FINAL APPROVED REPORT")
    print("------------------------")
    print(result["draft_report"])


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    run_autonomous_sales_agent("sales.xlsx")
