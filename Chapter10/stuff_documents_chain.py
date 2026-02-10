from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from transformers import pipeline
import warnings
import transformers

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
transformers.logging.set_verbosity_error()

# 1. Sample documents with metadata for filtering 
docs = [
    Document(
        page_content="Intermittent fasting improves insulin sensitivity.",
        metadata={"author": "Alice", "category": "health", "year": 2022}
    ),
    Document(
        page_content="The 2008 financial crisis impacted global markets.",
        metadata={"author": "Bob", "category": "finance", "year": 2008}
    ),
    Document(
        page_content="The French Revolution began in 1789.",
        metadata={"author": "Charles", "category": "history", "year": 1789}
    ),
]

# 2. Embed both content and metadata
def combine_metadata(doc: Document):
    meta_text = " ".join(f"{k}: {v}" for k, v in doc.metadata.items())
    return Document(page_content=f"{meta_text}. {doc.page_content}", metadata=doc.metadata)

docs_with_meta = [combine_metadata(doc) for doc in docs]

# 3. Create embeddings and vector store 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs_with_meta, embeddings)

# 4. Create LLM pipeline for text generation 
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1,   # CPU
    max_length=256
)

# 5. Wrap the pipeline in a LangChain LLM
llm = HuggingFacePipeline(pipeline=generator)

# 6. Create a retriever from the vector store 
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 7. Helper to generate LLM input and get response
def run_llm(context: str, question: str) -> str:
    """
    Build input prompt from context and question, then invoke the LLM
    """
    input_text = f"Context:\n{context}\nQuestion: {question}\nAnswer:"
    return llm.invoke(input_text)  # fixed: use invoke instead of run

# 8. RAG query function
def rag_query(query: str):
    """
    Retrieve documents, combine context, and run LLM
    """
    retrieved_docs = retriever._get_relevant_documents(run_manager=None, query=query)  # type: ignore # _get_relevant_documents needs run_manager
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    answer = run_llm(context_text, query)
    return answer, retrieved_docs

# 9. Example queries
queries = [
    "What did Alice write about health?",
    "Tell me about financial events after 2005",
    "History events before 1900"
]

# 10. Execute queries
for q in queries:
    ans, docs_used = rag_query(q)
    print(f"\nQuery: {q}")
    print("Answer:", ans)
    print("Documents used:")
    for d in docs_used:
        print(f"- {d.page_content}")
