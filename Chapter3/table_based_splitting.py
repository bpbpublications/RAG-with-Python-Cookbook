# table_based_splitting.py
# This code demonstrates how to split documents based on tabular data.
import pandas as pd
from langchain.schema import Document

# 1. Sample tabular data
# This data contains product information that we want to convert into documents.
# 2. Sample data is created as a pandas DataFrame.
data = pd.DataFrame({
    "Product": ["A", "B", "C"],
    "Description": ["Book A", "Book B", "Book C"]
})

# 3. Convert each row of the DataFrame into a Document
# Each document will contain the product name and description.
documents = [Document(page_content=f"{row.Product}: {row.Description}") for row in data.itertuples()]

# 3. Print the resulting documents
# Each document will represent a product with its description.
for i, doc in enumerate(documents):
    print(f"Chunk {i+1}:\n{doc.page_content}\n")
