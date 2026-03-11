import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import embeddings


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_retriever():
    docs = []

    folder = "crypto_docs"

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            text = load_text_file(path)

            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename}
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})