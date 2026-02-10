# This code is part of the LangChain framework and is used to load documents from various formats.
# It demonstrates how to load PDF, TXT, and DOCX files into a list of Document objects.
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

# Initialize list to store all documents which will be loaded from different formats
all_docs_list = []

# 1. Load PDF file and store it in all_docs_list where all_docs_list is a list of Document objects
# PyPDFLoader is used to load PDF files
pdf_loader = PyPDFLoader("RAG.pdf")

# pdf_doc is of type list of Document where each Document has page_content and metadata attributes
# If the PDF has multiple pages, it will return a list of Document objects, where each Document corresponds to a page in the PDF.
pdf_doc = pdf_loader.load()

# Extend the all_docs_list with the loaded PDF documents
# This will add the PDF content of the file as a Document object to the all_docs_list
all_docs_list.extend(pdf_doc)

# 2. Load TXT file and store it in all_docs_list list
# TextLoader is used to load text files
txt_loader = TextLoader("RAG.txt")

# TextLoader reads the text file and splits it into paragraphs
# Each paragraph will be treated as a separate Document object.
# txt_doc is of type list of Document where each Document has page_content and metadata attributes
# If the text file has multiple paragraphs, it will return a list of Document objects, where each Document corresponds to a paragraph in the text file.
txt_doc = txt_loader.load()

# Extend the all_docs_list with the loaded text documents
# This will add the text content of the file as a Document object to the all_docs_list
all_docs_list.extend(txt_doc)

# 3. Load DOCX file and store it in all_docs
# UnstructuredWordDocumentLoader is used to load DOCX files
# It is useful for loading Word documents that may contain complex formatting.
docx_loader = UnstructuredWordDocumentLoader("RAG.docx")

# docx_doc is of type list of Document where each Document has page_content and metadata attributes
# If the DOCX file has multiple sections, it will return a list of Document objects
docx_doc = docx_loader.load()

# Extend the all_docs_list with the loaded DOCX documents
# This will add the DOCX content of the file as a Document object to the all_docs_list
all_docs_list.extend(docx_doc)

# Print the total number of documents loaded 
print(f"\nTotal documents loaded: {len(all_docs_list)}\n")

# Print the content of the loaded three types of documents
# This will print the first 275 characters of each document's content
for i, doc in enumerate(all_docs_list[:]):  
    print(f"--- Document {i+1} ---")
    print(doc.page_content[:275])  
    print("\n")
