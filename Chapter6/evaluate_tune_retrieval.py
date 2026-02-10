# evaluate_tune_retrieval.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the document
loader = TextLoader("chapter6_sample_doc.txt")
docs = loader.load()

# 2. Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Build the vector store with specified chunk size and overlap
# This function will be used to create the vector store for evaluation
def build_vectorstore(chunk_size=300, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embedding_model)

# 4. Define evaluation queries
# These queries will be used to evaluate the retrieval performance
eval_queries = {
    "What is Artificial Intelligence?": "AI is the ability of machines to simulate human intelligence.",
    "Explain Deep Learning.": "Deep Learning uses neural networks with multiple layers.",
    "Applications of AI in healthcare.": "AI assists in disease diagnosis and personalized treatment."
}

# 5. Evaluate the vector store
# This function will run the evaluation using the vector store and the predefined queries
def evaluate(vectorstore, top_k=3):
    correct = 0
    total = len(eval_queries)

    for query, expected in eval_queries.items():
        results = vectorstore.similarity_search(query, k=top_k)
        retrieved_texts = " ".join([r.page_content for r in results])

        if expected.lower().split()[0] in retrieved_texts.lower():  # simple keyword match
            correct += 1

    return correct / total  # accuracy score

# 6. Tune parameters and evaluate
# This section will iterate over different chunk sizes, overlaps, and top_k values to find the best configuration
chunk_sizes = [200, 300, 500]
overlaps = [20, 50, 100]
top_ks = [2, 3, 5]

best_score = 0
best_params = None

for cs in chunk_sizes:
    for ov in overlaps:
        vs = build_vectorstore(chunk_size=cs, overlap=ov)
        for k in top_ks:
            score = evaluate(vs, top_k=k)
            print(f"Chunk={cs}, Overlap={ov}, top_k={k} → Accuracy={score:.2f}")
            if score > best_score:
                best_score = score
                best_params = (cs, ov, k)

# 7. Output the best parameters found
print("\nBest Parameters Found:")
print(f"Chunk Size={best_params[0]}, Overlap={best_params[1]}, top_k={best_params[2]} with Accuracy={best_score:.2f}") # type: ignore
