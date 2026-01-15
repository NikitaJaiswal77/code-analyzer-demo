import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PATH = os.path.join(BASE_DIR, "sample_project")  # folder with your Python code
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "rag", "vector_store")  # where FAISS index will be saved

def load_python_files(folder_path):
    """Load all .py files in the project as documents."""
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".py"):
            file_path = os.path.join(folder_path, file)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
    return documents

# Load all Python code files
docs = load_python_files(PROJECT_PATH)

# Split code into smaller chunks for better retrieval
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,     # max characters per chunk
    chunk_overlap=50   # overlap between chunks
)
chunks = splitter.split_documents(docs)

# Generate embeddings for the chunks
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create a FAISS vector store from the chunks
db = FAISS.from_documents(chunks, embeddings)

# Save the vector store to disk
os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
db.save_local(VECTOR_STORE_PATH)

print("✅ RAG: Code files indexed successfully")
