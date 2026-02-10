# ngram_splitter.py
# This module defines a custom text splitter that divides text into n-grams with overlap.
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, TokenTextSplitter

# n-gram text splitter class
class NGramTextSplitter(TextSplitter):
    # Initialize with n-gram size and overlap
    def __init__(self, n: int = 10, overlap: int = 2):
        super().__init__()
        self.n = n
        self.overlap = overlap

    # Split text into n-grams with specified overlap
    def split_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        step = self.n - self.overlap
        for i in range(0, len(words) - self.n + 1, step):
            chunk = " ".join(words[i:i + self.n])
            chunks.append(chunk)
        return chunks
    
    # Create LangChain-style documents from the text
    def create_documents(self, texts: List[str], metadata: List[dict] = None) -> List[Document]: # type: ignore
        documents = []
        metadata = metadata or [{} for _ in texts]
        for text, meta in zip(texts, metadata):
            splits = self.split_text(text)
            for chunk in splits:
                documents.append(Document(page_content=chunk, metadata=meta))
        return documents
