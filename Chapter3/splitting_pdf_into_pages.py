# splitting_pdf_into_pages.py
# This code demonstrates how to split a PDF into individual pages using LangChain's PyPDFLoader.
from langchain_community.document_loaders import PyPDFLoader

# 1. Load the PDF file
# Ensure the PDF file path is correct.
loader = PyPDFLoader("RAG_3pages.pdf")
docs = loader.load()

# 2. Print the number of pages loaded and a preview of each page's content
# This will help verify that the PDF has been split correctly into individual pages.
print(f"Total Pages: {len(docs)}")

# 3. Display the content of each page
# Here we print the first 100 characters of each page to give a preview.
for i, doc in enumerate(docs):
    print(f"\n--- Page {i+1} ---")
    print(doc.page_content[:100])  # print first 100 characters
