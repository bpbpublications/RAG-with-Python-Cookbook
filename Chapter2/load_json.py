# Load_json.py
# This script will load JSON document, which can be structured or unstructured,
# and print their content and metadata.
from langchain_community.document_loaders import JSONLoader

# 1. Load JSON documents from a file
# Modify the file path and jq_schema as needed
loader = JSONLoader(
    file_path="sample.json",
    jq_schema=".",  # "." means load the whole list of objects
    text_content=False  # We'll get structured documents instead of raw strings
)

# 2. Load the documents using the loader
# This will return a list of Document objects
docs = loader.load()

# 3. Print the loaded documents
# Each document will have a page content and metadata attributes
for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("---")
