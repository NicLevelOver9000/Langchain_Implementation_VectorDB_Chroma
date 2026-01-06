from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API keys
load_dotenv()

# Same embedding model used during indexing
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load EXISTING vector database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Prompt (required for LCEL)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a junior research assistant.
        Always answer ONLY using the context provided.
        You MUST answer ONLY using the context provided.
        If the answer is not in the context, then answer I don't know.""".strip()
    ),
    (
        "user",
        "Context:\n{context}"
    ),
    (
        "user",
        "Question:\n{question}"
    )
])
# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# RAG chain
rag_chain = (
    RunnableMap({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
)

# Ask a question
print(rag_chain.invoke("What is Java?"))
