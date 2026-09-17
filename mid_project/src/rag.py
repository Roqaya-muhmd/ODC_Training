from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_knowledge():
    """Load documents from the knowledge folder."""

    documents = []

    knowledge_path = Path("knowledge")

    for file in knowledge_path.glob("*.txt"):
        loader = TextLoader(
            str(file),
            encoding="utf-8"
        )

        documents.extend(loader.load())

    return documents

def create_vector_store():
    """Create FAISS vector store from knowledge documents."""

    documents = load_knowledge()

    print("Documents loaded:", len(documents))

    for doc in documents:
        print("Document content:", doc.page_content[:200])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    print("Chunks created:", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store