from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from transformers.pipelines import pipeline
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 2. Create a text generation pipeline using a small model for 
# demonstration purposes 
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=generator)

# 3. Create a vector store from the documents using 
# HuggingFace embeddings  
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create a simple dataset for demonstration purposes 
docs = [
    Document(page_content="The Eiffel Tower is located in Paris, France."),
    Document(page_content="The Great Wall of China is a historic fortification."),
    Document(page_content="The Colosseum is an ancient amphitheater in Rome."),
]

vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 5. Step-Back question generator prompt
step_back_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a reasoning assistant. Reformulate the user’s question into a broader, "
        "more general step-back question that captures its essence.\n\n"
        "Examples:\n"
        "Q: Who discovered penicillin?\n"
        "Step-back: What are important discoveries in the history of medicine?\n\n"
        "Q: When was the iPhone first released?\n"
        "Step-back: What are major product launch events in technology history?\n\n"
        "Q: {question}\n"
        "Step-back:"
    ),
)
step_back_chain = step_back_prompt | llm

# 6. Create a QAprompt  
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use the following context to answer the user’s question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)
qa_chain = qa_prompt | llm

# --- Orchestration function ---
def step_back_rag(question: str):
    # 7. Reformulate the original query
    step_back_question = step_back_chain.invoke({"question": question}).strip()

   	# 8. Retrieve the document using step-back question
    retrieved_docs = retriever.invoke(step_back_question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Generate final answer
    final_answer = qa_chain.invoke({"context": context, "question": question}).strip()

    return {
        "original_question": question,
        "step_back_question": step_back_question,
        "retrieved_context": context,
        "final_answer": final_answer,
    }

# --- Run example ---
if __name__ == "__main__":
    query = "Where is the Eiffel Tower located?"
    result = step_back_rag(query)
#9. Print
    print("\n=== Step-Back Chain Execution ===")
    print("Original Question:", result["original_question"])
    print("Step-Back Reformulation:", result["step_back_question"])
    print("Retrieved Context:", result["retrieved_context"])
    print("Final Answer:", result["final_answer"])
