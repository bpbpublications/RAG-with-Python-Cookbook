# html_tag_based_splitting.py
# This code demonstrates how to split HTML content into chunks using LangChain's text splitter.
# It extracts text from HTML tags and splits it into manageable chunks.
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Sample HTML document
# This HTML contains various tags that we want to extract and split into chunks.
html_doc = """
<html>
  <body>
    <h1>RAG Pipeline</h1>
    <p>Load → Split → Embed → Retrieve → Generate</p>
    <p>Used in chatbots, document search, and more.</p>
  </body>
</html>
"""

# 2. Parse the HTML and extract text from specific tags
# Here, we will extract text from <h1> and <p> tags.
soup = BeautifulSoup(html_doc, "html.parser")
paragraphs = soup.find_all(["h1", "p"])
documents = [Document(page_content=tag.get_text()) for tag in paragraphs]

# 3. Split the extracted text into chunks
# Using RecursiveCharacterTextSplitter to split the documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_documents(documents)

# 4. Print the resulting chunks
# Each chunk will retain the text extracted from the HTML tags.
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")
