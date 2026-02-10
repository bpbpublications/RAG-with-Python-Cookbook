# custom_metadata_during_loading.py
# Custom Metadata During Document Loading
# This script demonstrates how to add custom metadata to documents loaded from a text file using LangChain
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# 1. Load a text file using LangChain's TextLoader
loader = TextLoader("RAG.txt")

# 2. Load the documents from the text file using the loader
# This will create a loader instance and load the content of the file
raw_docs = loader.load()

# 3. Add custom metadata to each document
# Custom metadata can be any key-value pair you want to associate with the document
# In this example this includes source, category, author, etc.
custom_docs = []
for doc in raw_docs:
    doc.metadata["source"] = "local_file"
    doc.metadata["category"] = "tutorial"
    doc.metadata["author"] = "Deepak"
    custom_docs.append(doc)

# 4. Print the content and metadata of the loaded documents
# Each document will have the custom metadata added 
for i, doc in enumerate(custom_docs):
    print(f"\n--- Document {i+1} ---")
    print("Content:", doc.page_content[:105])
    print("Metadata:", doc.metadata)
