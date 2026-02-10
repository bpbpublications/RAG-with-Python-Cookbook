#load_markdown_file.py
# Load a Markdown file using LangChain's UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader


# 1. Specify the path to your Markdown file
# Replace 'sample_markdown_file.md' with the actual path to your Markdown file, ir required
# Ensure the file exists in the specified location
file_path = "sample_markdown_file.md"

# 2. Load the Markdown file
# This will create a loader instance and load the content of the file
# The UnstructuredMarkdownLoader will parse the Markdown file and convert it into a list of documents
loader = UnstructuredMarkdownLoader(file_path)
docs = loader.load()

# 3. Print the content of the loaded documents
# Each document corresponds to a section in the Markdown file
for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---\n")
    print(doc.page_content)
