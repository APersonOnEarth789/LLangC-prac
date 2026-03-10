from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# 1. Load your knowledge base
# -----------------------------
def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


bitcoin_text = load_text_file("bitcoin.txt")
ethereum_text = load_text_file("ethereum.txt")
defi_text = load_text_file("defi.txt")

docs = [
    Document(page_content=bitcoin_text, metadata={"source": "bitcoin.txt"}),
    Document(page_content=ethereum_text, metadata={"source": "ethereum.txt"}),
    Document(page_content=defi_text, metadata={"source": "defi.txt"}),
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -----------------------------
# 2. Define tools
# -----------------------------
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


# -----------------------------
# 3. Create model
# -----------------------------
llm = ChatOllama(
    model="qwen3.5",
    temperature=0,
)


# -----------------------------
# 4. Create agent
# -----------------------------
system_prompt = """
You are a crypto research assistant.

Your job:
- Answer user questions clearly and accurately.
- Use the search_knowledge_base tool whenever the question asks about Bitcoin, Ethereum, DeFi, blockchain concepts, or anything factual from the knowledge base.
- If tool results are available, base your answer only on those results.
- If the answer is not in the tool results, say you are not sure.
- Be concise but informative.
- When possible, mention the source file used.
"""

agent = create_agent(
    model=llm,
    tools=[search_knowledge_base, list_topics],
    system_prompt=system_prompt,
)


# -----------------------------
# 5. Run interactive loop
# -----------------------------
if __name__ == "__main__":
    print("Crypto Agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )

        messages = response["messages"]
        final_message = messages[-1].content
        print(f"\nAgent: {final_message}\n")