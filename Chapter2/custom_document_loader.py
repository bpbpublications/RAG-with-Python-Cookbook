# custom_document_loader.py
# This script defines a custom document loader for loading text files
# It uses LangChain's BaseLoader to create a custom loader that reads text files
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
import os

# 2. CustomTextLoader is a class that extends BaseLoader to load text files
# It reads the content of a text file and returns it as a Document object
class CustomTextLoader(BaseLoader):
    def __init__(self, file_path: str, metadata: dict = None): # type: ignore
        self.file_path = file_path
        self.metadata = metadata or {}

    def load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(page_content=text, metadata=self.metadata)]

# 1. Main is executed when the script is run directly
# It creates an instance of CustomTextLoader with a specified file path and metadata
if __name__ == "__main__":
    file_path = "RAG.txt"
    custom_metadata = {
        "source": file_path,
        "category": "session_notes",
        "author": "Deepak",
        "tags": ["custom", "demo", "loader"]
    }

    loader = CustomTextLoader(file_path=file_path, metadata=custom_metadata)
    docs = loader.load()

# 3. Print the loaded documents
# Each document will have its content and metadata printed to the console
    for doc in docs:
        print("Text Content:\n", doc.page_content)
        print("\nMetadata:\n", doc.metadata)
