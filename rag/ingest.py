import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PATH = os.path.join(BASE_DIR, "sample_project")  
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "rag", "vector_store")  

def load_python_files(folder_path):
    """Load all .py files in the project as documents."""
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".py"):
            file_path = os.path.join(folder_path, file)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
    return documents

docs = load_python_files(PROJECT_PATH)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    
    chunk_overlap=100, 
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = FAISS.from_documents(chunks, embeddings)


os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
db.save_local(VECTOR_STORE_PATH)

print(" RAG: Code files indexed successfully")
