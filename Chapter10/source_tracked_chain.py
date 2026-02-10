from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


# --------------------------------------
# 1. Initialize LLM
# --------------------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=256,
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=generator)


# --------------------------------------
# 2. Embeddings
# --------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------
# 3. Documents WITH SOURCES
# --------------------------------------
docs = [
    Document(
        page_content="Intermittent fasting improves metabolic health.",
        metadata={"source": "doc1"}
    ),
    Document(
        page_content="The 2008 financial crisis impacted global markets.",
        metadata={"source": "doc2"}
    ),
    Document(
        page_content="The French Revolution began in 1789 and reshaped Europe.",
        metadata={"source": "doc3"}
    ),
]


# --------------------------------------
# 4. Vectorstore + Retriever
# --------------------------------------
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


# --------------------------------------
# 5. Prompt Template (Source-Aware Answering)
# --------------------------------------
answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Answer ONLY using the context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)


# --------------------------------------
# 6. The Source-Aware Chain Function
# --------------------------------------
def source_aware_chain(question: str):

    # Step 1: Retrieve documents
    retrieved_docs = retriever.invoke(question)

    # Step 2: Combine context text
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # Step 3: Generate answer via LCEL
    answer = (answer_prompt | llm).invoke({
        "context": context_text,
        "question": question
    }).strip()

    # Step 4: Extract sources
    sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]

    return {
        "query": question,
        "answer": answer,
        "sources": sources
    }


# --------------------------------------
# 7. Test Queries (like your example)
# --------------------------------------
queries = [
    "What did Alice write about health?",
    "Tell me about financial events after 2005",
    "History events before 1900"
]


# --------------------------------------
# 8. Run the chain and print output
# --------------------------------------
for q in queries:
    result = source_aware_chain(q)
    print("\nQuery:", result["query"])
    print("Answer: Content:", result["answer"])
    print("Sources:", result["sources"])
