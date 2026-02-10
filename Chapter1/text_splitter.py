# It demonstrates how to split a document into smaller chunks using LangChain's text splitter.
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. Load the document
# Replace 'RAG.txt' with your actual text file path.
# Ensure the file exists in the specified path.
# This will load the text file into a list of Document objects.
loader = TextLoader("RAG.txt")  # Replace with your actual file
documents = loader.load()

# 2. Initialize the text splitter
# This will split the text into chunks based on specified parameters.
text_splitter = CharacterTextSplitter(
    separator="\n",       # Splits at newline characters
    chunk_size=300,       # Max characters per chunk
    chunk_overlap=50,     # Overlap to preserve context
    length_function=len   # Optional, default is len()
)

# 3. Split the documents into chunks
# This will create smaller text segments for better processing and retrieval.
split_docs = text_splitter.split_documents(documents)

# 4. Output the results
# This will print each chunk to the console.
for i, doc in enumerate(split_docs):
    print(f"\n== Chunk {i+1} ==")
    print(doc.page_content)
