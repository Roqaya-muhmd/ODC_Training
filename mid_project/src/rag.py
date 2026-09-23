from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_knowledge():
    """Load documents (TXT + PDF) from the knowledge folder."""

    documents = []

    knowledge_path = Path("knowledge")

    for file in knowledge_path.glob("*.txt"):
        loader = TextLoader(
            str(file),
            encoding="utf-8"
        )
        documents.extend(loader.load())

    for file in knowledge_path.glob("*.pdf"):
        loader = PyPDFLoader(str(file))
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


def add_dataset_to_store(vector_store, summary_text, dataset_name="uploaded_dataset"):
    """
    Adds an uploaded dataset's structured summary text into the SAME FAISS
    vector store as the PDFs/TXT knowledge base — so retrieval stays unified
    instead of routing dataset questions through a separate pandas path.
    """

    document = Document(
        page_content=summary_text,
        metadata={"source": f"Uploaded Dataset Summary ({dataset_name})"}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents([document])

    print(f"Adding {len(chunks)} dataset-summary chunks to the vector store...")

    vector_store.add_documents(chunks)

    return vector_store