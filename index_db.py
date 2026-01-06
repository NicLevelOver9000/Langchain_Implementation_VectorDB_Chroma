from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Example data (replace with DB rows, files, APIs, etc.)
data = [
    "Java is an object-oriented programming language.",
    "Spring Boot is used to build production-ready Java applications.",
    "JVM allows Java programs to run on multiple platforms."
]

documents = [
    Document(page_content=text)
    for text in data
]

# Embedding model (must match main.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create + persist vector database
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ Vector database created and populated.")
print("Stored documents:", vectorstore._collection.count())
