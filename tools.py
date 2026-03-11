from langchain.tools import tool
from rag import build_retriever

retriever = build_retriever()


@tool
def search_knowledge_base(question: str) -> str:
    """Search the crypto knowledge base for relevant information."""
    results = retriever.invoke(question)

    if not results:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk = doc.page_content.strip()
        formatted.append(f"[{i}] Source: {source}\n{chunk}")

    return "\n\n".join(formatted)


@tool
def list_topics(_: str = "") -> str:
    """List the topics currently available in the knowledge base."""
    return "Available topics: Bitcoin, Ethereum, DeFi"