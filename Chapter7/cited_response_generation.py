# cited_response_generation.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers.pipelines import pipeline
import json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the document
loader = TextLoader("chapter7_RAG.txt")   # replace with your docs
docs = loader.load()

# 2. Split the documents into chunks of 500 characters with 50 character overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

# 3. Create embeddings and vector store from documents using HuggingFaceEmbeddings and FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. Create a text generation pipeline using a local model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=256
)

# 5. Define function cited_response(query: str) that takes a query string as input
# and returns a structured response with the answer and sources of retrieved documents
def cited_response(query: str):
    # retrieve top documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)

    # build context with numbered references
    numbered_context = ""
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", f"doc{i}")
        numbered_context += f"[{i}] ({source}): {doc.page_content}\n"

    # prompt instructing the model to cite sources
    prompt = f"""Question: {query}
Use the following references to answer, and cite them like [1], [2].

References:
{numbered_context}

Answer with citations:
"""

    # generate answer
    answer = generator(prompt, max_length=256)[0]["generated_text"]

    # return structured result
    return {
        "question": query,
        "answer": answer,
        "sources": [
            {"id": i+1, "source": doc.metadata.get("source", "unknown"), "content": doc.page_content[:200] + "..."}
            for i, doc in enumerate(retrieved_docs)
        ]
    }

# 6. Query which will use cited response generation to generate an answer
query = "What is Retrieval-Augmented Generation?"

# 7. Generate the response using cited_response function
response = cited_response(query)

# 8. Print the structured response in JSON format
print(json.dumps(response, indent=2))