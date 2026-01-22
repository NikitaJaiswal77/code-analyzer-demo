import sys
sys.stdout.reconfigure(line_buffering=True)

import yaml
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.prompts import PromptTemplate

with open("rag/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

EXPLAIN_LEVEL = CONFIG["analysis"].get("verbosity", "simple")    
MODE = CONFIG["analysis"].get("mode", "learning")               
FUNCTION_NAMES = ["login", "logout", "helper"]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "rag/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=80,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a Python code assistant.\n"
        "Explain code behavior in words.\n"
        "Do NOT repeat code.\n"
        "If a function has only `pass`, say it is not implemented.\n\n"
        "Code:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

def explain_function_from_context(context, function_name):
    if f"def {function_name}(" not in context:
        return None

    lines = context.splitlines()
    inside = False
    body = []

    for line in lines:
        if line.strip().startswith(f"def {function_name}"):
            inside = True
            continue
        if inside:
            if line.strip().startswith("def "):
                break
            if line.strip():
                body.append(line.strip())

    if not body or body == ["pass"]:
        return f"The `{function_name}` function is defined but not implemented yet."

    if EXPLAIN_LEVEL == "simple":
        return f"The `{function_name}` function contains logic."
    else:
        return f"The `{function_name}` function prints a message or performs actions as defined in the code."

def find_unimplemented_functions(context):
    functions = []
    lines = context.splitlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            name = line.split("(")[0].replace("def", "").strip()
            if i + 1 < len(lines) and lines[i + 1].strip() == "pass":
                functions.append(name)
    return functions

print("Ask questions about your project code (type 'exit')")

while True:
    query = input("Question: ").strip()

    if not query:
        print(" Please ask a code-related question.\n")
        continue

    if query.lower() == "exit":
        print(" Exiting chat.")
        break

    print(" Thinking...")
    result = qa.invoke({"query": query})
    docs = result.get("source_documents", [])
    context = "\n".join(doc.page_content for doc in docs)

    if "project" in query.lower():
        print(
            "\nAnswer: This project analyzes a Python codebase using a "
            "Retrieval-Augmented Generation system. It indexes source code, "
            "detects unimplemented functions, and explains code behavior in natural language.\n"
        )
        continue

    if "not implemented" in query.lower():
        funcs = find_unimplemented_functions(context)
        if funcs:
            print("\nAnswer: Unimplemented functions:")
            for f in funcs:
                print("-", f)
            print()
        else:
            print("\nAnswer: All functions are implemented.\n")
        continue

    function_name = None
    for name in FUNCTION_NAMES:
        if name in query.lower():
            function_name = name
            break

    if function_name:
        explanation = explain_function_from_context(context, function_name)
        if explanation:
            print("\nAnswer:", explanation, "\n")
            continue

    answer = result["result"].strip()
    if not answer:
        print("\nAnswer: No relevant code found.\n")
    else:
        print("\nAnswer:", answer, "\n")
