# custom_separator_keywords_splitter.py
# This code splits a text into chunks based on custom keywords "START" and "END".
from langchain_core.documents import Document

# 1. Define a custom text with specific keywords to split on.
# The text is structured such that it contains multiple chunks
text = "START This is first chunk. END START This is a second chunk. END"

# 2. Split the text into chunks using the custom keywords
# and create Document objects for each chunk.
# The text is split at "START" and then each chunk is processed
# to extract content up to the next "END" keyword.
chunks = text.split("START")
documents = []
for chunk in chunks:
    if "END" in chunk:
        content = chunk.split("END")[0].strip()
        documents.append(Document(page_content=content))

# 3. Print the resulting Document objects
# Each Document represents a chunk of text between the custom keywords.
for i, doc in enumerate(documents, 1):
    print(f"[Chunk {i}] {doc.page_content}")
