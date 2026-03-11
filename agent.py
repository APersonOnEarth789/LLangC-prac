from langchain.agents import create_agent
from config import llm
from tools import search_knowledge_base, list_documents, get_crypto_price, retrieve_topic_info

system_prompt = """
You are a crypto research assistant.

You can answer questions about cryptocurrency, blockchain technology, and decentralized finance.

You have access to several tools:

search_knowledge_base  
Use this tool when the user asks about crypto concepts such as Bitcoin, Ethereum, DeFi, mining, staking, smart contracts, or blockchain technology.

get_crypto_price  
Use this tool when the user asks for the current price of a cryptocurrency.

list_documents  
Use this tool when the user asks what topics or documents are available in the knowledge base.

summarize_topic  
Use this tool when the user asks for a summary of a crypto topic.

Rules:
- When answering, use short paragraphs or bullet points if helpful.
- Prefer using tools instead of answering from memory.
- If information comes from the knowledge base, mention the source file.
- If the knowledge base does not contain the answer, say you are not sure.
- Be clear, concise, and informative.
"""

agent = create_agent(
    model=llm,
    tools=[
        search_knowledge_base,
        list_documents,
        get_crypto_price,
        retrieve_topic_info
    ],
    system_prompt=system_prompt,
)

if __name__ == "__main__":
    print("Crypto Agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("Ask me a question: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )

        messages = response["messages"]
        final_message = messages[-1].content
        print(f"\nAgent: {final_message}\n")