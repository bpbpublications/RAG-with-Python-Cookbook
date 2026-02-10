# confidence_scored_response.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers.pipelines import pipeline
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load and split the document
loader = TextLoader("chapter7_RAG.txt")   # Replace with your document
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 2. Create embeddings and FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings)

generator = pipeline("text2text-generation", model="google/flan-t5-base")

# 3. Define confidence scoring function
def get_confidence(scores):
    """Normalize similarity scores to a 0–1 confidence scale."""
    if not scores:
        return 0.0
    # FAISS similarity is cosine distance → lower = closer
    # Convert to similarity by inverting
    sims = [1 - s for s in scores]
    sims = np.clip(sims, 0, 1)  # ensure valid range
    return float(np.mean(sims))

# 4. Query with confidence scoring
query = "Explain what Retrieval-Augmented Generation (RAG) is."

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)

# 5. Extract context and scores
context = "\n".join([doc.page_content for doc, _ in docs_with_scores])
scores = [score for _, score in docs_with_scores]
confidence = get_confidence(scores)

# 6. Generate response using context and query       
prompt = f"Context:\n{context}\n\nQuestion: {query}"
response = generator(prompt, num_return_sequences=1)

# 7. Output response and confidence score
print("\n--- RESPONSE ---")
print(response[0]['generated_text'])
print("\n--- CONFIDENCE SCORE ---")
print(f"{confidence:.2f}")
