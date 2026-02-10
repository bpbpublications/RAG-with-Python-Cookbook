# chunk_search_rerank.py
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch

nltk.download('punkt')

# 1. Sample documents for demonstration purposes 
# These can be longer documents that will be chunked into smaller pieces
DOCUMENTS = [
    """
    Retrieval-Augmented Generation (RAG) is a method that combines retrieval and generation.
    It reduces hallucinations by grounding answers in retrieved documents.
    Dense retrieval uses vector embeddings for semantic similarity.
    BM25 is a sparse retrieval method based on keyword frequency.
    Vector databases store embeddings for fast similarity search.
    """,
    """
    Cooking recipes require precise measurements and timing.
    Ingredients must be prepared in advance.
    Following instructions carefully ensures good results.
    """
]

# Chunking Strategy: Split documents into overlapping chunks of words
def chunk_document(doc, chunk_size=30, overlap=5):
    """
    Split a document into word-based chunks with overlap.
    """
    words = doc.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# 2. Chunk all documents and create a flat list of chunks
all_chunks = []
for doc in DOCUMENTS:
    all_chunks.extend(chunk_document(doc, chunk_size=30, overlap=5))

# 3. Initialize SentenceTransformer model for embeddings
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)
chunk_embeddings = model.encode(all_chunks, convert_to_tensor=True, normalize_embeddings=True)

# 4. Perform semantic search to find top-k relevant chunks
def semantic_search(query, top_k=3):
    query_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(query_emb, chunk_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(all_chunks)))
    return [(all_chunks[idx], float(score)) for score, idx in zip(top_results.values, top_results.indices)]

# 5. Initialize Cross-Encoder for reranking
from sentence_transformers import CrossEncoder
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(cross_encoder_model)

# 6. Re-rank the top-k chunks using Cross-Encoder
def rerank(query, candidates):
    pairs = [(query, cand) for cand in candidates]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return reranked

if __name__ == "__main__":
    # Example Query
    query = "How does RAG reduce hallucinations?"

    search_results = semantic_search(query, top_k=5)
    # 7. Display top chunks from semantic search with scores
    print("Top Chunks from Semantic Search:")
    for chunk, score in search_results:
        print(f"  score={score:.4f} | {chunk}")

    chunks_only = [chunk for chunk, _ in search_results]
    reranked_results = rerank(query, chunks_only)

    # 8. Display reranked results with scores
    print("\nReranked Results:")
    for chunk, score in reranked_results:
        print(f"  score={score:.4f} | {chunk}")
