# Load_html.py
# Load a web page using LangChain's BSHTMLLoader
# This example assumes you have a local HTML file named 'sample.html'
from langchain_community.document_loaders import BSHTMLLoader

# 1. Specify the path to your HTML file
# You can replace 'sample.html' with the actual path to your HTML file
# Get loader instance using BSHTMLLoader
loader = BSHTMLLoader("sample.html")

# 2. Load the HTML file using the loader
# This will create a loader instance and load the content of the file
docs = loader.load()

# 3. Print the content of the loaded documents
# Each document corresponds to a section in the HTML file
print(f"Loaded {len(docs)} HTML documents")
print(docs[0].page_content[:500])
