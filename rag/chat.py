import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.prompts import PromptTemplate

# -------------------------------
# Paths
# -------------------------------
VECTOR_STORE_PATH = "rag/vector_store"

# -------------------------------
# Load embeddings and FAISS store
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# -------------------------------
# HuggingFace Pipeline
# -------------------------------
hf_pipeline = pipeline(
    "text-generation",  # use causal LM-compatible task
    model="google/flan-t5-base",  # your model
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    device=-1  # -1 for CPU, change to 0 if using GPU
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# -------------------------------
# Prompt template
# -------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful AI assistant that summarizes Python project code.\n"
        "Do NOT repeat or quote the code. Only explain what the project does.\n"
        "Answer in 2–3 simple sentences.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
)

# -------------------------------
# RetrievalQA chain
# -------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt}
)

# -------------------------------
# Chat loop
# -------------------------------
print("Ask questions about your project code (type 'exit')\n")

while True:
    query = input("Question: ").strip()
    if query.lower() == "exit":
        break

    try:
        # Ask the QA chain directly
        response = qa.invoke({"query": query})
        print("\nAnswer:", response["result"], "\n")
    except Exception as e:
        print("\n Error:", e, "\n")
