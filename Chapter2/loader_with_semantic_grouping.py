# loader_with_semantic_grouping.py
# This script demonstrates how to load a document using LangChain's UnstructuredLoader
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

# 1. Specify the path to your file. This can be a text file, PDF, DOCX, HTML, etc.
# Ensure the file exists in the specified path.
file_path = "Unstructured RAG.txt" 

# 2. Load the document using UnstructuredFileLoader
# This loader will automatically handle the file format and extract the content.
loader = UnstructuredLoader(file_path)
docs = loader.load()

# 3. Output the loaded documents
# Each document will have metadata and content, which can be used for further processing.
for i, doc in enumerate(docs):
    print(f"\n Section {i+1}")
    print("Metadata:", doc.metadata)
    print("Content Preview:\n", doc.page_content[:300], "\n---")
