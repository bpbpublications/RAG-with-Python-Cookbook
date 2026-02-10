from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from langchain_core.prompts import PromptTemplate

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 1. Load the document
loader = TextLoader("RAG.txt")   # Make sure 'RAG.txt' exists
docs = loader.load()

# 2. Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

# 3. Create vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. Initialize LLM (local)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=256,
    device=-1  # CPU
)
llm = HuggingFacePipeline(pipeline=generator)

# 5. Build prompt template
prompt = PromptTemplate.from_template(
    "Use the context below to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)

# 6. Define RAG function
def rag_chain(query: str, vectorstore, llm, k: int = 3):
    """
    RAG chain for latest LangChain:
    1. Retrieve top-k relevant documents from vectorstore
    2. Combine into context
    3. Format prompt
    4. Call LLM to generate answer
    """
    # Retrieve top-k documents directly from vectorstore
    docs = vectorstore.similarity_search(query, k=k)

    # Combine documents into a single context string
    context = "\n\n".join([d.page_content for d in docs])

    # Format the prompt
    prompt_text = prompt.format(context=context, question=query)

    # Generate answer using LLM
    return llm.invoke(prompt_text)

# 7. Run a query
query = "What is Retrieval-Augmented Generation?"
answer = rag_chain(query, vectorstore, llm)

# 8. Print the result
print("Question:", query)
print("Answer:", answer)
