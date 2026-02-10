# adaptive_chunking_for_different_content_type.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. Load sample document
loader = TextLoader("chapter6_adaptive_chunking_sample_doc.txt")
docs = loader.load()

# 2. Define smart chunking logic based on content type
# This function will adaptively chunk text based on its structure
def smart_chunking(doc_content: str):
    chunks = []
    blocks = doc_content.split("\n\n")  # split by paragraphs / code blocks

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if block.startswith("def ") or block.startswith("class "):
            # Code → keep whole block intact
            chunks.append(block)

        elif "-" in block:
            # Bullets → keep together unless too long
            if len(block) > 400:
                splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                chunks.extend(splitter.split_text(block))
            else:
                chunks.append(block)

        else:
            # Narrative text → allow larger chunks
            if len(block) > 500:
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks.extend(splitter.split_text(block))
            else:
                chunks.append(block)

    return chunks


# 3. Apply smart chunking to all documents
# This will create adaptive chunks based on content type
all_chunks = []
for doc in docs:
    chunks = smart_chunking(doc.page_content)
    all_chunks.extend(chunks)

# 4. Output the results
print(f"Total chunks created: {len(all_chunks)}\n")
for i, chunk in enumerate(all_chunks, start=1):
    print(f"Chunk {i}:\n{chunk}\n{'-'*50}")
