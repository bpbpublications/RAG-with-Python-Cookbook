# Load and split a DOCX document into smaller chunks
# This example uses the UnstructuredWordDocumentLoader to load a DOCX file,
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# and the RecursiveCharacterTextSplitter to split it into manageable chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1️. Load the DOCX document
docx_loader = UnstructuredWordDocumentLoader("RAG.docx") 
# The loader reads the DOCX file and returns a list of Document objects.
# Each Document object contains the text content and metadata of a page or section. 
documents = docx_loader.load()                            

# 2️. Display basic stats about the loaded document
splitter = RecursiveCharacterTextSplitter(
    chunk_size     = 300,   # max characters (≈ 120‑150 tokens) per chunk
    chunk_overlap  = 50,    # characters of overlap to preserve context
    separators     = ["\n\n", "\n", ".", " ", ""],  # split on these characters
)
    
# 3️. Split the document into smaller chunks
chunks = splitter.split_documents(documents)

# 4️. Display the number of chunks created and preview 
print(f"\nTotal chunks created: {len(chunks)}\n")
for i, chunk in enumerate(chunks):        #iterate through each chunk
    # Print the chunk number and its character count
    print(f"--- Chunk {i+1} ({len(chunk.page_content)} chars) ---")
    print(chunk.page_content.strip()[:300])   # preview first 300 chars
    print()
