# Load_web_page.py
# This script demonstrates how to load a web page using LangChain's WebBaseLoader
# It fetches the content from a specified URL and prints the first 1000 characters of
from langchain_community.document_loaders import WebBaseLoader

# 1. Get Loader instance using WebBaseLoader
# Specify the URL of the web page you want to load
loader = WebBaseLoader("https://example.com/")

# 2. Load the web page using the loader
# This will create a loader instance and load the content of the web page
docs = loader.load()

# 3. Print the 1000 characters from the content
# Each document corresponds to the content of the web page
for i, doc in enumerate(docs):
    print(f"--- Document {i+1} ---")
    print(doc.page_content[:1000])  # Print first 1000 characters
    print("\nMetadata:", doc.metadata)
