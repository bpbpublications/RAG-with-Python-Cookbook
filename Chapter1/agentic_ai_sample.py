# Install required packages if not already installed
# pip install langchain==1.2 transformers torch pandas

from json import tool
import pandas as pd
import pandas as pd
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.llms import HuggingFaceHub
from transformers import pipeline



# -------- Load Hugging Face model -------- #
# Using a text-generation model (you can change to any HF model)
hf_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_length=256
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# -------- Define tools for the agent -------- #
def load_sales_data(file_path: str) -> str:
    """Load CSV sales data and return as string."""
    df = pd.read_csv(file_path)
    return df.to_string()

def calculate_revenue(file_path: str) -> str:
    """Calculate total revenue from CSV."""
    df = pd.read_csv(file_path)
    total_revenue = df["revenue"].sum()
    return f"Total Revenue: ₹{total_revenue}"

def analyze_trends(file_path: str) -> str:
    """Analyze revenue trends by region."""
    df = pd.read_csv(file_path)
    region_sales = df.groupby("region")["revenue"].sum()
    top_region = region_sales.idxmax()
    return f"Top region by revenue: {top_region} with ₹{region_sales[top_region]}"

# Wrap functions as LangChain tools
tools = [
    Tool(name="Load Sales Data", func=load_sales_data, description="Load CSV sales data"),
    Tool(name="Calculate Revenue", func=calculate_revenue, description="Sum up revenue from CSV"),
    Tool(name="Analyze Trends", func=analyze_trends, description="Analyze revenue trends by region")
]

# -------- Initialize the agent -------- #
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -------- Example usage -------- #
# Make sure you have a CSV file like "sales.csv" with columns: region,revenue
file_path = "sales.csv"
response = agent.run(f"Calculate revenue from {file_path} and analyze trends.")
print(response)
