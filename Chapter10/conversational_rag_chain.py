# conversational_rag_chain.py
# Example of a conversational RAG chain using LangChain with a local HuggingFace model and FAISS vector store.
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from transformers.pipelines import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Sample documents to populate the vector store 
docs = [
    Document(page_content="RAG reduces hallucinations by grounding answers in retrieved evidence."),
    Document(page_content="Dense retrieval uses vector embeddings instead of keyword matching."),
    Document(page_content="BM25 is a sparse retrieval method based on term frequency statistics."),
    Document(page_content="Vector databases store embeddings to enable semantic search."),
]

# 2. Create vector store and retriever with FAISS and HuggingFace embeddings 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

# 3. Load local HuggingFace model (e.g., Flan-T5) and create LLM wrapper 
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 4. Create ConversationalRetrievalChain with components 
# ConversationalRetrievalChain manages chat history internally hence no custom prompt needed
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

# 5. Maintain chat history and run multiple queries  
chat_history = []

# First query
query1 = "How does RAG reduce hallucinations?"

# 6. Run the first query through the chain and print the answer
result1 = qa_chain.invoke({"question": query1, "chat_history": chat_history})
print(f"Q1: {query1}")
print("A1:", result1["answer"])
chat_history.append((query1, result1["answer"]))

# Follow-up query (depends on context)
query2 = "And what method does it use for retrieval?"

# 7. Run the follow-up query through the chain and print the answer 
result2 = qa_chain.invoke({"question": query2, "chat_history": chat_history})
print(f"\nQ2: {query2}")
print("A2:", result2["answer"])
