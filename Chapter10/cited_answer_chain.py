# cited_answer_chain.py
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline
import faiss

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# 1. Sample documents with sources with mix of relevant and irrelevant content
docs = [
    {"content": "Intermittent fasting improves insulin sensitivity.", "source": "Source 1"},
    {"content": "It may help with weight loss by reducing calorie intake.", "source": "Source 2"},
    {"content": "It supports autophagy, the body’s cell repair process.", "source": "Source 3"},
    # Irrelevant documents
    {"content": "Paris is the capital of France.", "source": "Irrelevant 1"},
    {"content": "Python is a programming language for AI and data science.", "source": "Irrelevant 2"},
]

# 2. Embedding and retrieval setup using FAISS for simplicity
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embs = embedder.encode([d["content"] for d in docs])

index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs) # type: ignore

# 4. Retrieve top-k documents above similarity threshold for a query
def retrieve(query, k=5, threshold=0.5):
    q_emb = embedder.encode([query])
    distances, idxs = index.search(q_emb, k) # type: ignore
    
    results = []
    for dist, idx in zip(distances[0], idxs[0]):
        sim = 1 / (1 + dist)  # crude similarity from L2
        if sim >= threshold:
            results.append(docs[idx])
    return results

# 3. Load a text generation model for answering with citations 
gen = pipeline("text2text-generation", model="google/flan-t5-base")

def cited_answer(query, k=5, threshold=0.5):
    retrieved = retrieve(query, k, threshold)
    
    if not retrieved:
        return "No relevant information found.", []
    
    # 5. Construct context with sources for generation 
    context = "\n".join([f"{d['content']} ({d['source']})" for d in retrieved])
    
    # 6. Prompt the model to answer with citations 
    prompt = f"""
Answer the question using ALL the relevant context facts below. 
List each fact as a separate bullet point and always include its source in parentheses. 
Do not skip or merge facts. Ignore irrelevant content.

Question: {query}
Context:
{context}

Answer:"""
    
    # 7. Generate the answer with citations 
    out = gen(prompt, max_new_tokens=200)[0]["generated_text"]
    
    if "(" not in out or len(out.splitlines()) < len(retrieved):
        out = "\n".join([f"- {d['content']} ({d['source']})" for d in retrieved])
    
    return out

if __name__ == "__main__":
    query = "What are the benefits of intermittent fasting?"
    k = 5  # number of docs to retrieve
    
    answer = cited_answer(query, k)
    
    # 8. Display the answer with citations     
    print("\n=== Cited Answer Chain Output ===")
    print(answer)
