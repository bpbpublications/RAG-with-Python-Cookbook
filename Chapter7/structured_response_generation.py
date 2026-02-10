from typing import Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
import warnings, json

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# -------------------------------
# 1. Load documents
# -------------------------------
loader = TextLoader("chapter7_RAG.txt")
docs = loader.load()

# -------------------------------
# 2. Split documents
# -------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

# -------------------------------
# 3. Vector store
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# -------------------------------
# 4. LLM
# -------------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=256,
    device=-1  # CPU
)
llm = HuggingFacePipeline(pipeline=generator)

# -------------------------------
# 5. Prompt template
# -------------------------------
prompt = PromptTemplate.from_template(
    "Use the context below to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)

# -------------------------------
# 6. RAG function
# -------------------------------
def structured_answer(query: str) -> Dict[str, Any]:
    # Retrieve top 3 relevant documents
    docs = vectorstore.similarity_search(query, k=3)

    # Combine content for context
    context = "\n\n".join([d.page_content for d in docs])

    # Format prompt
    prompt_text = prompt.format(context=context, question=query)

    # Generate answer
    answer_text = llm.invoke(prompt_text)

    # Gather sources
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    # Estimate confidence
    confidence = round(min(1.0, len(answer_text) / 300), 2)

    return {
        "question": query,
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence
    }

# -------------------------------
# 7. Run query
# -------------------------------
query = "What is Retrieval-Augmented Generation?"
response = structured_answer(query)

# -------------------------------
# 8. Print structured response
# -------------------------------
print(json.dumps(response, indent=2))
