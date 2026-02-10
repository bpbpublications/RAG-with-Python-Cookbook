from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Sample documents to populate the vector store
docs = [
    Document(page_content="RAG reduces hallucinations by grounding answers in retrieved evidence."),
    Document(page_content="Dense retrieval uses vector embeddings instead of keyword matching."),
]

# 2. Create vector store and retriever with FAISS and HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(docs, embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 2})

# 3. Load local HuggingFace model (e.g., Flan-T5) and create LLM wrapper
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 4. Define a simple prompt formatter
def format_prompt(context: str, question: str) -> str:
    return f"""
Use the following context to answer the question. 
If you don't know the answer, just say "I don't know".

Context:
{context}

Question: {question}
Answer:
"""

# 5. Run a query
query = "How does RAG reduce hallucinations?"

# Retrieve top documents
retrieved_docs = retriever._get_relevant_documents(query, run_manager=None) # type: ignore

# Concatenate context
context_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Prepare prompt
input_prompt = format_prompt(context_text, query)

# Generate answer using HuggingFacePipeline
# In LangChain 1.0.5, use .generate()
answer_obj = llm.generate([input_prompt])
answer_text = answer_obj.generations[0][0].text

# Print results
print(f"Query: {query}")
print(f"Answer: {answer_text}")
